"""One additional pre-decision time, bounded SELECT only; no source update."""
from collections import Counter,defaultdict
from datetime import date,timedelta
import json,math
import numpy as np
from ..core import RACE_KEY,SafeAuditError
from ..db_connection import validate_audit_sql
from ..ml import safety,market
from ..research import io,experiment as exp
from ..win_sim import core,publication,local
from .. import stage1,stage1_analysis as selection
from . import contracts as c,store as s,data


def query(races,starts):
    if not 1<=len(races)<=32 or len(set(races))!=len(races):raise safety.MLFailure('input_limit')
    ident=safety.identifier;values=[];params=[]
    for race in races:
        day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));c.guard(day,day)
        if starts[race].date()!=day:raise safety.MLFailure('invalid_date')
        bound=starts[race]-timedelta(minutes=61)
        values.append('('+','.join(['%s::text']*7)+')');params.extend((*race,bound.strftime('%m%d%H%M')))
    keys=','.join(ident(k) for k in RACE_KEY);equality=' AND '.join('s.'+ident(k)+'::text=w.'+ident(k) for k in RACE_KEY)
    selected=[f'w.{ident(k)}' for k in RACE_KEY]
    for alias,(field,width) in publication.O1_FIELDS.items():
        selected.append(f'CASE WHEN pg_catalog.octet_length(t.{ident(field)})<={width} THEN t.{ident(field)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('WITH w('+keys+',bound) AS (VALUES '+','.join(values)+') SELECT '+','.join(selected)+
        ', EXISTS(SELECT 1 FROM ONLY public.apd_sokuho_o1 s WHERE '+equality+
        " AND (s.data_kubun::text NOT IN ('1','2','3','4','5') OR "
        "(s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text !~ '^[0-9]{8}$'))) AS unsafe_version "
        'FROM w LEFT JOIN LATERAL (SELECT s.* FROM ONLY public.apd_sokuho_o1 s WHERE '+equality+
        " AND s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text<=w.bound "
        'ORDER BY s.happyo_tsukihi_jifun DESC LIMIT 2) t ON true')
    validate_audit_sql(sql);return sql,tuple(params)


def decode_rows(raw):
    grouped=defaultdict(list)
    for row in raw:
        race=tuple(row[k] for k in RACE_KEY)
        if row['record_type'] is not None:
            item={k:v for k,v in row.items() if k not in (*RACE_KEY,'unsafe_version')}
            item.update(race_key=race,source_table='apd_sokuho_o1',oversized=False);grouped[race].append(item)
        if row['unsafe_version']:grouped[race].append(dict(race_key=race,status='0',source_table='apd_sokuho_o1'))
    return grouped


def delta_features(packet,quote60):
    earlier=packet['quotes']['30']
    if earlier is None:return None,'T30_unavailable'
    if quote60 is None:return None,'T60_unavailable'
    q30=stage1._quote(earlier);q60=stage1._quote(quote60)
    if set(q30.odds)!=set(q60.odds):return None,'roster_mismatch'
    elapsed=(q30.published_at-q60.published_at).total_seconds()/60
    if not elapsed>0:return None,'nonpositive_publication_elapsed'
    p30,_=market.implied_probabilities(q30.odds);p60,_=market.implied_probabilities(q60.odds)
    return {h:{c.EXTRA[0]:math.log(float(p30[h]))-math.log(float(p60[h])),c.EXTRA[1]:elapsed,c.EXTRA[2]:0} for h in p30},None


def load(directory,start,end):
    c.guard(start,end);out={}
    for year in range(start.year,end.year+1):
        folder=directory/'T60';done=s.completed(folder,str(year))
        if not done:raise safety.MLFailure('artifact_invalid')
        for name in done['summary']['batch_files']:
            for item in s.read(folder,name)['records']:
                race=tuple(item['race_key']);day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));c.guard(day,day)
                if start<=day<=end:
                    if race in out:raise safety.MLFailure('duplicate_entry')
                    out[race]=item
    return out


def augment(directory,frame):
    if not len(frame):return frame.copy()
    start,end=date.fromisoformat(str(frame.date.min())),date.fromisoformat(str(frame.date.max()))
    packets=data.market(directory,start,end);extra=load(directory,start,end);values={}
    for race,packet in packets.items():
        if packet['quotes']['10'] is None:continue
        if race not in extra:raise safety.MLFailure('artifact_invalid')
        fields,reason=delta_features(packet,extra[race]['quote'])
        if fields:
            for h,row in fields.items():values[''.join(race)+h]=row
    result=frame.copy()
    for name in c.EXTRA:result[name]=[values.get(str(k),{}).get(name,1 if name==c.EXTRA[2] else np.nan) for k in frame.entry_key]
    if result.entry_key.tolist()!=frame.entry_key.tolist():raise safety.MLFailure('invalid_identity')
    return result


def run(directory):
    if not s.completed(directory,'reproduce'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'quotes60')
    if done:return done['summary']
    folder=directory/'T60';folder.mkdir(exist_ok=True);catalog=[];files=[]
    for year in range(2020,2026):
        prior=s.completed(folder,str(year))
        if prior:catalog.append(prior['summary']);files.append(folder/f'{year}.complete.json');continue
        start,end=date(year,1,1),min(c.END,date(year,12,31));f=data.market_frame(directory,start,end);packets=data.market(directory,start,end)
        rosters={p.race:tuple(p.probabilities) for p in exp.predictions(f,f.market_probability.to_numpy())};keys=sorted(rosters)
        batch_files=[];reason_counts=Counter();usable=0;requested=0;returned=0;executed=0;elapsed=[];ages=[]
        for offset in range(0,len(keys),32):
            name=f'{year}-{offset//32:03d}.local.json';batch_files.append(name);batch=keys[offset:offset+32]
            if (folder/name).exists():result=s.read(folder,name)
            else:
                starts={r:core.instant(packets[r]['scheduled_start']) for r in batch}
                valid=tuple(r for r in batch if packets[r]['quotes']['30'] is not None)
                raw=[];error=None
                if valid:
                    try:
                        with io.session() as db:
                            columns=publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,local.Budget(120))
                            if columns is None:raise safety.MLFailure('invalid_schema')
                            sql,params=query(valid,starts);raw=db.aggregate('s4.T60_asof',sql,params,max_rows=256)
                    except SafeAuditError as exc:error=exc.code
                grouped=decode_rows(raw);records=[]
                for race in batch:
                    quote=None;reason='T30_unavailable'
                    if race in valid:
                        schedule=core.Schedule(race,starts[race],None,rosters[race],True,'reviewed_historical_schedule')
                        if error:reason='T60_read_'+error
                        else:quote,reason=selection.select_quote(race,grouped[race],rosters[race],schedule,60,300)
                    value=stage1._quote_document(quote);fields,delta_reason=delta_features(packets[race],value)
                    records.append(dict(race_key=race,quote=value,reason=reason if quote is None else delta_reason,
                        scheduled_start=starts[race].isoformat(),T30_published_at=packets[race]['quotes']['30']['published_at'] if packets[race]['quotes']['30'] else None,
                        elapsed_T60_to_T30_minutes=next(iter(fields.values()))[c.EXTRA[1]] if fields else None,
                        T60_age_seconds=(starts[race]-timedelta(minutes=60)-quote.published_at).total_seconds() if quote else None,
                        processing_basis='JRA_publication_posthoc_not_PC_observation',schedule_asof_verified=False))
                result={'records':records,'requested_races':len(valid),'rows_returned':len(raw),'query_executed':bool(valid),
                    'query_error':error,'source_table':'public.apd_sokuho_o1','max_rows':256,'batch_races_max':32,'statement_timeout_ms':60000}
                s.write(folder,name,result)
            requested+=result['requested_races'];returned+=result['rows_returned'];executed+=result['query_executed']
            for item in result['records']:
                if item['elapsed_T60_to_T30_minutes'] is not None:
                    usable+=1;elapsed.append(item['elapsed_T60_to_T30_minutes']);ages.append(item['T60_age_seconds'])
                else:reason_counts[item['reason'] or 'history_unavailable']+=1
        summary={'year':year,'T10_primary_races':len(keys),'usable_T60_T30_races':usable,
            'usable_rate_percent':100*usable/len(keys) if keys else None,'missing_reasons':dict(reason_counts),
            'requested_races':requested,'rows_returned':returned,'bounded_queries':executed,
            'elapsed_minutes':core.distribution(elapsed),'age_seconds':core.distribution(ages),'batch_files':batch_files}
        s.complete(folder,str(year),[folder/n for n in batch_files],summary);catalog.append(summary);files.append(folder/f'{year}.complete.json')
        print(json.dumps(dict(status='running',stage='quotes60',**{k:v for k,v in summary.items() if k!='batch_files'})),flush=True)
    result={'years':[{k:v for k,v in row.items() if k!='batch_files'} for row in catalog],
        '2026_rows_read':0,'new_history_timepoints':[60],'all_T10_primary_races_retained':True,'no_DB_mutations':True}
    s.complete(directory,'quotes60',files,result);return result
