"""Bounded index lookups for latest pre-cutoff O1 versions, never final prices."""
from collections import Counter,defaultdict
from datetime import date,timedelta
import json
import math

from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..ml import market,safety
from ..win_sim import core,local,publication,settle_saved
from .. import stage1_analysis as s1a,stage1
from . import contracts as c,io


def asof_query(races,starts):
    if not 1<=len(races)<=32:raise safety.MLFailure('input_limit')
    ident=safety.identifier
    values=[];params=[]
    for race in races:
        day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
        c.guard_development(day,day)
        if day<date(2020,1,1):raise safety.MLFailure('invalid_split')
        for offset in (30,10):
            cutoff=starts[race]-timedelta(minutes=offset+1)
            values.append('('+','.join(['%s::text']*6+['%s::integer','%s::text'])+')')
            params.extend((*race,offset,cutoff.strftime('%m%d%H%M')))
    keys=','.join(ident(k) for k in RACE_KEY)
    equality=' AND '.join('s.'+ident(k)+'::text=w.'+ident(k) for k in RACE_KEY)
    selections=[f'w.{ident(k)}' for k in RACE_KEY]+['w.minutes']
    for alias,(field,width) in publication.O1_FIELDS.items():
        selections.append(f'CASE WHEN pg_catalog.octet_length(t.{ident(field)})<={width} THEN t.{ident(field)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('WITH w('+keys+',minutes,bound) AS (VALUES '+','.join(values)+') SELECT '+','.join(selections)+
         ', EXISTS(SELECT 1 FROM ONLY public.apd_sokuho_o1 s WHERE '+equality+
         " AND (s.data_kubun::text NOT IN ('1','2','3','4','5') OR "
         "(s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text !~ '^[0-9]{8}$'))) AS unsafe_version "
         'FROM w LEFT JOIN LATERAL (SELECT s.* FROM ONLY public.apd_sokuho_o1 s WHERE '+equality+
         " AND s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text<=w.bound "
         'ORDER BY s.happyo_tsukihi_jifun DESC LIMIT 2) t ON true')
    validate_audit_sql(sql)
    return sql,tuple(params)


def collect(directory):
    from .experiment import load_cache,key_parts
    from ..post_step2 import confirmed_hr_source
    folder=safety.local_path(directory/'market',root=c.ROOT);folder.mkdir(exist_ok=True)
    catalog=[];total=Counter();sources=[]
    for year in range(2020,2026):
        start=date(year,1,1);end=min(c.END,date(year,12,31))
        frame=load_cache(directory,start,end)
        grouped=defaultdict(list)
        for row in frame[['entry_key','start_time','previous_start']].to_dict('records'):
            parts=key_parts(str(row['entry_key']));grouped[parts[:-1]].append(row)
        keys=sorted(grouped);out=[]
        for offset in range(0,len(keys),32):
            path=folder/f'{year}-{offset//32:03d}.local.json'
            if path.exists():
                existing=io.read(folder,path.name);out.extend(existing['records']);continue
            batch=keys[offset:offset+32];schedules={};reasons={};starts={}
            for race in batch:
                first=grouped[race][0]
                def plain(v):return None if v is None or str(v)=='<NA>' else str(v)
                roster=tuple(key_parts(str(r['entry_key']))[-1] for r in grouped[race])
                ra={'record_type':'RA','status':'7','start':plain(first['start_time']),
                    'previous_start':plain(first['previous_start']),'oversized':False}
                schedule,reason=s1a.reference_schedule(race,[ra],roster)
                schedules[race]=schedule;reasons[race]=reason
                if schedule:starts[race]=schedule.scheduled_start
            valid=tuple(race for race in batch if schedules[race])
            raw=[];outcomes=[]
            with io.session() as db:
                budget=local.Budget(120)
                columns=publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,budget)
                if columns is None:raise safety.MLFailure('invalid_schema')
                if valid:
                    sql,params=asof_query(valid,starts)
                    raw=db.aggregate('s2.market_asof',sql,params,max_rows=256)
                settling=tuple(r for r in batch if int(r[0]) in (2023,2024,2025) and
                    (r[1]<='0930' or int(r[0])==2025))
                if settling:
                    source=confirmed_hr_source(settling)
                    hr,source,_=settle_saved.fetch_hr(db,settling,max_rows=200,reviewed_source=source)
                    outcomes=settle_saved.decode_existing_hr(hr,source)
            by_target=defaultdict(list)
            for row in raw:
                race=tuple(row[k] for k in RACE_KEY);minutes=row['minutes']
                if row['record_type'] is not None:
                    item={k:v for k,v in row.items() if k not in (*RACE_KEY,'minutes','unsafe_version')}
                    item.update(race_key=race,source_table='apd_sokuho_o1',oversized=False)
                    by_target[(race,minutes)].append(item)
                if row['unsafe_version']:
                    by_target[(race,minutes)].append(dict(race_key=race,status='0',source_table='apd_sokuho_o1'))
            records=[]
            omap={o.race:o for o in outcomes}
            for race in batch:
                roster=tuple(key_parts(str(r['entry_key']))[-1] for r in grouped[race])
                item={'race_key':race,'scheduled_start':starts[race].isoformat() if race in starts else None,
                      'schedule_asof_verified':False,'quotes':{},'failures':{},'raw_selected_O1':{}}
                for minutes in (30,10):
                    rows=by_target[(race,minutes)]
                    q,reason=s1a.select_quote(race,rows,roster,schedules[race],minutes) if schedules[race] else (None,reasons[race])
                    item['quotes'][str(minutes)]=stage1._quote_document(q)
                    item['failures'][str(minutes)]=reason
                    item['raw_selected_O1'][str(minutes)]=rows
                o=omap.get(race)
                item['outcome']=dict(status=o.status,payouts_yen_per_100=o.payouts_yen_per_100,
                    verified=o.verified,complete_winner_set=o.complete_winner_set,reason=o.reason) if o else None
                records.append(item)
            io.write(folder,path.name,{'records':records,'read_completed_at_jst':stage1.now().isoformat()})
            out.extend(records)
        summary={'year':year,'target_races':len(out),'primary_usable':sum(r['quotes']['10'] is not None for r in out),
            'early_usable':sum(r['quotes']['30'] is not None for r in out),
            'primary_exclusions':dict(Counter(r['failures']['10'] for r in out if not r['quotes']['10'])),
            'history_exclusions':dict(Counter(r['failures']['30'] for r in out if not r['quotes']['30']))}
        catalog.append(summary)
        print(json.dumps(dict(status='running',stage='market',**summary)),flush=True)
    manifest={'years':catalog,'contract':io.read(directory,'plan.json')['quote_contract'],
              'source':'apd_sokuho_o1_intermediate_only_latest_index_lookup','sealed_2026_rows':0}
    io.write(directory,'market_manifest.json',manifest)
    return manifest


def load(directory,start,end):
    c.guard_development(start,end)
    result={}
    for year in range(max(2020,start.year),end.year+1):
        for path in sorted((directory/'market').glob(f'{year}-*.local.json')):
            for row in io.read(path.parent,path.name)['records']:
                race=tuple(row['race_key']);day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
                if start<=day<=end:
                    if race in result:raise safety.MLFailure('duplicate_entry')
                    result[race]=row
    return result


def augment(frame,packets):
    import numpy as np
    import pandas as pd
    from .experiment import key_parts
    mapped={};stats=Counter()
    for race,row in packets.items():
        current=row['quotes']['10'];early=row['quotes']['30']
        if current is None:continue
        q=stage1._quote(current);pm,overround=market.implied_probabilities(q.odds);ranks=market._average_ranks(pm)
        pe=eranks=None
        if early is not None:
            earlier=stage1._quote(early);pe,_=market.implied_probabilities(earlier.odds)
            if set(pe)==set(pm):eranks=market._average_ranks(pe)
            else:pe=None
        elapsed=(core.instant(current['published_at'])-core.instant(early['published_at'])).total_seconds()/60 if pe is not None else None
        for h in pm:
            mapped[(''.join(race)+h)]={'market_probability':float(pm[h]),'market_rank':ranks[h],
                'market_log_odds':math.log(float(q.odds[h])),'market_inverse_sum':float(overround),
                'market_log_probability_change':math.log(float(pm[h]))-math.log(float(pe[h])) if pe is not None else None,
                'market_rank_improvement':eranks[h]-ranks[h] if pe is not None else None,
                'market_elapsed_minutes':elapsed,'market_history_missing':int(pe is None)}
    out=frame.copy()
    for field in c.MARKET_FEATURES+c.HISTORY_FEATURES:
        out[field]=[mapped.get(str(k),{}).get(field,np.nan) for k in frame['entry_key']]
    # Whole-race matching was already verified by the O1 decoder. Early missingness is never a main-cohort filter.
    return out.loc[out['market_probability'].notna()].copy()
