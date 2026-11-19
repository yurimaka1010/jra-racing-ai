"""Exclusive local artifacts and bounded development-only SELECT adapters."""
from collections import Counter, defaultdict
from datetime import date, timedelta
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from ..core import RACE_KEY
from ..db_connection import read_only_session, validate_audit_sql
from ..inspect_schema import COLUMNS_SQL
from ..ml import safety
from ..win_sim import core, local, publication
from ..win_sim_cli import connection_config
from . import contracts as c


def write(directory,name,value):
    safety.write_json(directory/name,value,root=c.ROOT)


def read(directory,name):
    return safety.read_json(directory/name,root=c.ROOT)


def session():
    if not os.environ.get('PGPASSWORD'):
        raise safety.MLFailure('needs_authentication')
    return read_only_session(connection_config(60000),os.environ['PGPASSWORD'])


def months(start=c.START,end=c.END):
    c.guard_development(start,end)
    cursor=start
    while cursor<=end:
        next_month=date(cursor.year+(cursor.month==12),1 if cursor.month==12 else cursor.month+1,1)
        yield cursor,min(end,next_month-timedelta(days=1))
        cursor=next_month


def source_query(columns,start,end):
    c.guard_development(start,end)
    from ..ml.safety import identifier as ident
    selections=['ra.'+ident(k)+'::text AS '+ident(k) for k in RACE_KEY]
    for alias,(table,name,width,role) in c.RAW_FIELDS.items():
        if name not in columns[table]:
            if alias in ('horse_number','horse_id','finish_position','surface','distance','declared_runners','frame','age','sex','weight_raw'):
                raise safety.MLFailure('invalid_schema')
            selections.append('NULL::text AS '+ident(alias));continue
        ref=table+'.'+ident(name)
        selections.append(f'CASE WHEN pg_catalog.octet_length({ref})<={width} THEN {ref}::text ELSE NULL::text END AS '+ident(alias))
    join=' AND '.join('ra.'+ident(k)+'=se.'+ident(k) for k in RACE_KEY)
    sql=('SELECT '+','.join(selections)+' FROM ONLY public.jvd_ra ra JOIN ONLY public.jvd_se se ON '+join+
         " WHERE ra.data_kubun::text='7' AND se.data_kubun::text='7' AND ra.kaisai_nen::text=%s "
         'AND ra.kaisai_tsukihi::text BETWEEN %s AND %s AND ra.keibajo_code::text IN '+
         "('01','02','03','04','05','06','07','08','09','10') ORDER BY "+
         ','.join('ra.'+ident(k) for k in RACE_KEY)+',se.'+ident('umaban')+' LIMIT %s')
    validate_audit_sql(sql)
    return sql,(str(start.year),start.strftime('%m%d'),end.strftime('%m%d'),20001)


def collect(directory):
    from ..ml.data import TABLE_SQL
    output=safety.local_path(directory/'source',root=c.ROOT);output.mkdir(exist_ok=True)
    columns={};manifest=[];counts=Counter();missing=Counter();anomalies=Counter()
    with session() as db:
        for alias in ('ra','se'):
            info=db.aggregate('s2.table',TABLE_SQL,('public','jvd_'+alias),max_rows=1)
            if (len(info)!=1 or info[0]['kind']!='r' or info[0]['access_method']!='heap'
                    or any(info[0][k] for k in ('is_partition','descendants','rls','forced_rls'))):
                raise safety.MLFailure('invalid_schema')
            raw=db.aggregate('s2.columns',COLUMNS_SQL,('public','jvd_'+alias),max_rows=1000)
            columns[alias]={x['name']:x for x in raw}
        mapping={name:{'table':'jvd_'+t,'column':physical,'max_bytes':width,'role':role,
            'present':physical in columns[t], 'source_type':columns[t].get(physical,{}).get('data_type')}
            for name,(t,physical,width,role) in c.RAW_FIELDS.items()}
        for t,cols in columns.items():
            for name in [*RACE_KEY,*[p for _,(alias,p,_,_) in c.RAW_FIELDS.items() if alias==t]]:
                if name in cols and (not cols[name]['type_is_builtin'] or cols[name]['type_name'] not in local.TEXT_TYPES):
                    raise safety.MLFailure('invalid_schema')
    write(directory,'mapping.v2.json',dict(format_version=2,input_format=c.INPUT_VERSION,fields=mapping,
        official_reference=str(local.SPEC_PDF),definition='JV RA/SE pages11/12, code tables2003/2007/2101/2102',
        units={'weight_raw':'.1kg','actual_runners':'actual_starters_past_only','late3f_raw':'.1seconds_000_999_missing_obstacle_is_1F_not_used'},
        margin='primary_adjacent_gap_not_winner_gap;plus_for_disqualification_not_summed',
        condition='age_specific_raw_categories;005/010/016 renamed win classes;no_numeric_order',
        temporal='seven_day_delay;final_snapshot_versions_not_verified_original_receipt'))
    for start,end in months():
        path=output/(start.strftime('%Y%m')+'.csv')
        if path.exists():
            # Completed monthly exports can be reused after a bounded interrupted collection.
            side=read(output,path.stem+'.json')
            if side['sha256']!=safety.digest(path,root=c.ROOT):raise safety.MLFailure('artifact_invalid')
            manifest.append(side);continue
        sql,params=source_query(columns,start,end)
        rows=0;local_missing=Counter();local_anomalies=Counter()
        with session() as db, path.open('x',encoding='utf-8',newline='') as stream:
            fields=(*RACE_KEY,*c.RAW_FIELDS)
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for batch in db.iter_batches('s2.source',sql,params,max_rows=20000,batch_size=1000,max_seconds=180):
                for row in batch:
                    # The source remains raw; transformations occur only under the versioned feature contract.
                    writer.writerow({k:'' if row.get(k) is None else row[k] for k in fields})
                    rows+=1
                    local_missing.update(k for k in c.RAW_FIELDS if row.get(k) is None or str(row[k]).strip()=='')
                    code=row.get('anomaly');local_anomalies[code if code in tuple('01234567') else 'unknown']+=1
        side={'file':path.name,'rows':rows,'start':start.isoformat(),'end':end.isoformat(),
              'sha256':safety.digest(path,root=c.ROOT),'raw_missing':dict(local_missing),'anomaly_counts':dict(local_anomalies)}
        write(output,path.stem+'.json',side);manifest.append(side)
        if start.month in (1,7):
            print(json.dumps(dict(status='running',stage='collect',through=start.strftime('%Y-%m'),rows_so_far=sum(x['rows'] for x in manifest))),flush=True)
    for item in manifest:
        counts['rows']+=item['rows'];missing.update(item['raw_missing']);anomalies.update(item['anomaly_counts'])
    result={'months':manifest,'total_rows':counts['rows'],'raw_missing':dict(missing),'anomaly_counts':dict(anomalies),
            'source_end':c.END.isoformat(),'sealed_2026_rows_read':0,'population':'all_JRA_01_10_RA7_SE7_not_target_horse_filtered'}
    write(directory,'source_manifest.json',result)
    return {k:v for k,v in result.items() if k!='months'}


def iter_days(directory):
    from itertools import groupby
    from .features import num
    manifest=read(directory,'source_manifest.json')
    for item in manifest['months']:
        path=safety.local_path(directory/'source'/item['file'],root=c.ROOT,exists=True)
        if safety.digest(path,root=c.ROOT)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        with path.open(encoding='utf-8',newline='') as stream:
            for day,rows in groupby(csv.DictReader(stream),key=lambda r:r['kaisai_nen']+r['kaisai_tsukihi']):
                result=[]
                for row in rows:
                    weight=num(row.get('weight_raw'))
                    row['weight_kg']=str(weight/10) if weight is not None else ''
                    result.append(row)
                yield result


def init(directory):
    from .. import stage1
    from ..stage1_analysis import variants,measure,IDENTITY
    from ..win_sim import core
    base=c.ROOT.parent
    preserved={}
    roots=['ml-v01/models','ml-step2/evaluations','post-step2/evaluations','stage1-market/evaluations',
           'trial-20261017/evaluations','trial-20261018/evaluations','ml-v02/simulations']
    for rel in roots:
        root=base/rel
        for path in root.rglob('*') if root.exists() else ():
            if path.is_file() and (path.suffix in ('.json','.md') or path.name.endswith('.lgb.txt') or path.name in ('predictions.csv','top_predictions.csv','learning_curve.csv')):
                preserved[str(path)]=safety.digest(path,root=base)
    write(directory,'preserved.json',preserved)
    plan=c.plan();plan['created_at_jst']=stage1.now().isoformat()
    plan['feature_contracts']={g:c.spec(tuple(g) if g!='control' else ()) for g in ('control','A','B','C','D','AB','ABCD')}
    write(directory,'plan.json',plan)
    prior=base/'stage1-market/evaluations/53fe500d9063471eb0cacd16b0bec85a'
    stage1.verify_preserved(prior,stage1.read(prior,'plan.json'))
    saved=stage1.read(prior/'analyses/e685462eea85407f90b85dc7124c9c28','results.json')
    records=stage1.read(prior,'predictions.local.json');packets,_=stage1.load_inputs(prior)
    maximum=0.
    for period in ('development','confirmation','reused_final'):
        selected=[r for r in records if r['period']==period and packets[tuple(r['race_key'])]['quotes']['10']]
        ai=[core.PredictionRace(tuple(r['race_key']),r['probabilities']) for r in selected]
        quotes=[stage1._quote(packets[tuple(r['race_key'])]['quotes']['10']) for r in selected]
        winners={tuple(r['race_key']):r['winner'] for r in selected}
        for name,pred in variants(ai,quotes,saved['selection']).items():
            actual=measure(pred,winners);expected=saved['periods'][period]['metrics'][name]
            maximum=max(maximum,*[abs(actual[k]-expected[k]) for k in ('log_loss','brier_score','race_log_loss','top1_accuracy_percent')])
    if maximum>1e-12:raise safety.MLFailure('artifact_invalid')
    result={'old_artifact_files':len(preserved),'stage1_reproduction_max_difference':maximum,
            'old_model_T075_unchanged':True,'sealed_2026_evaluated':False}
    write(directory,'reproduction.json',result)
    return result


def verify_preserved(directory):
    values=read(directory,'preserved.json');base=c.ROOT.parent
    different=sum(safety.digest(Path(path),root=base)!=expected for path,expected in values.items())
    if different:raise safety.MLFailure('artifact_invalid')
    return {'checked_files':len(values),'changed_old_artifacts':0,'sealed_2026_evaluated':False}
