"""Strict as-of adapter; existing H features/decoder, never fitting or labels."""
from collections import defaultdict
from datetime import timedelta
import pandas as pd
from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..ml import safety,market
from ..research import market_data,model
from ..win_sim import core,formats,publication
from .. import stage1
from . import contracts as c

def query(races,starts,offsets=(3,),*,historical=True):
    if not 1<=len(races)<=32 or len(set(races))!=len(races) or offsets not in ((3,),(30,10)):
        raise safety.MLFailure('input_limit')
    ident=safety.identifier;values=[];params=[]
    for race in races:
        if historical:c.historical(race)
        elif race[0]+race[1]<'20261022':raise safety.MLFailure('invalid_split')
        start=core.instant(starts[race])
        if start.strftime('%Y%m%d')!=race[0]+race[1]:raise safety.MLFailure('invalid_identity')
        for minutes in offsets:
            d=start-timedelta(minutes=minutes);low=d-timedelta(seconds=120 if minutes==3 else 300)
            values.append('('+','.join(['%s::text']*6+['%s::integer','%s::text','%s::text'])+')')
            params.extend((*race,minutes,low.strftime('%m%d%H%M'),(d-timedelta(minutes=1)).strftime('%m%d%H%M')))
    keys=','.join(ident(k) for k in RACE_KEY);equal=' AND '.join('s.'+ident(k)+'::text=w.'+ident(k) for k in RACE_KEY)
    cols=[f'w.{ident(k)}' for k in RACE_KEY]+['w.minutes']
    for alias,(field,width) in publication.O1_FIELDS.items():
        cols.append(f'CASE WHEN pg_catalog.octet_length(t.{ident(field)})<={width} THEN t.{ident(field)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('WITH w('+keys+',minutes,low,bound) AS (VALUES '+','.join(values)+') SELECT '+','.join(cols)+
        ', EXISTS(SELECT 1 FROM ONLY public.apd_sokuho_o1 s WHERE '+equal+
        " AND (s.data_kubun::text NOT IN ('1','2','3','4','5') OR (s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text !~ '^[0-9]{8}$'))) AS unsafe_version "+
        'FROM w LEFT JOIN LATERAL (SELECT s.record_id,s.data_kubun,s.happyo_tsukihi_jifun,s.hatsubai_flag_tansho,s.odds_tansho FROM ONLY public.apd_sokuho_o1 s WHERE '+equal+
        " AND s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text>=w.low AND s.happyo_tsukihi_jifun::text<=w.bound "+
        'ORDER BY s.happyo_tsukihi_jifun DESC LIMIT 2) t ON true')
    validate_audit_sql(sql);return sql,tuple(params)

def grouped(raw):
    out=defaultdict(list)
    for row in raw:
        race=tuple(row[k] for k in RACE_KEY);key=(race,row['minutes'])
        if row['record_type'] is not None:
            out[key].append({k:row.get(k) for k in publication.O1_FIELDS}|{'race_key':list(race),
                'source_table':'apd_sokuho_o1','oversized':False,'unsafe_version':bool(row['unsafe_version'])})
        elif row['unsafe_version']:out[key].append({'race_key':list(race),'status':'0','unsafe_version':True})
    return out

def select(race,rows,roster,cutoff,max_age,*,require_seen=False):
    eligible=[]
    for row in rows:
        if core.key(row['race_key'])!=tuple(race):return None,'roster_mismatch'
        if require_seen:
            seen=core.instant(row['first_seen_at'])
            if seen>cutoff:continue
        if row.get('unsafe_version') or row.get('status')!='1':return None,'version_unverified'
        try:published=formats.parse_time(row.get('published'),'mmddhhmm',race)
        except core.SimulationError:return None,'odds_time_unverified'
        if published+timedelta(minutes=1)>cutoff:continue
        if require_seen and published>seen:return None,'odds_observed_before_publication'
        eligible.append((published,row))
    if not eligible:return None,'no_on_time_observation' if require_seen else 'T3_history_unavailable'
    latest=max(t for t,_ in eligible);chosen=[row for t,row in eligible if t==latest]
    # Do not fall back to an older good row when the latest is conflicting/broken.
    if len(chosen)!=1:return None,'odds_conflict'
    if not 0<=(cutoff-latest).total_seconds()<=max_age:return None,'odds_stale'
    row=chosen[0]
    try:odds,cancelled=market.prices(row,tuple(roster))
    except core.SimulationError as exc:return None,exc.code
    # Existing chooser retains price-range/capping validation.
    snap=core.Snapshot(race,latest,core.instant(row['first_seen_at']) if require_seen else None,
        odds,row.get('source_table','apd_sokuho_o1'),True,publication_precision='minute')
    selected,reason=core.choose_snapshot(race,[snap],cutoff,roster,timing_basis='observed' if require_seen else 'jra_publication_time')
    if reason:return None,reason
    q=market.MarketRace(race,selected.odds,selected.source,'1','pre_race_publication_time',latest,
        cancelled_outside_roster=cancelled)
    document=stage1._quote_document(q)
    if require_seen:document['first_seen_at']=row['first_seen_at']
    return document,None

def features(race,current,early):
    roster=tuple(sorted(current['odds']))
    if early and set(early['odds'])!=set(roster):early=None
    f=pd.DataFrame({'entry_key':[''.join(race)+h for h in roster],
        'date':[f'{race[0]}-{race[1][:2]}-{race[1][2:]}']*len(roster)})
    result=market_data.augment(f,{tuple(race):{'quotes':{'10':current,'30':early}}})
    if len(result)!=len(roster):raise safety.MLFailure('invalid_identity')
    return result

def predict(bundle,race,current,early):
    f=features(race,current,early);p=model.predict(bundle,f)
    return {'H':dict(zip(f.entry_key.str[-2:],map(float,p))),
        'raw_market_T10':dict(zip(f.entry_key.str[-2:],map(float,f.market_probability)))},f
