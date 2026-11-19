"""Same-record O1 place price adapter. No price is interpreted as probability."""
from collections import Counter,defaultdict
from datetime import date
import json,math,re
import numpy as np
from ..research import io
from ..win_sim import local,publication,core
from ..ml import safety,market
from ..core import RACE_KEY,SafeAuditError
from ..db_connection import validate_audit_sql
from .. import stage1,stage1_analysis
from ..stage4 import data as old_data
from ..research import experiment as exp
from . import contracts as c,store as s

SPEC=c.ROOT.parent/'references/JRA-VAN/JV-Data4901.pdf'
SPEC_SHA='b6c21aae4ccbba6a71c5e8065609c4fbb1ccee826c16e7d99ca6ecf7a4101522'
FIELDS=dict(publication.O1_FIELDS,place_sale=('hatsubai_flag_fukusho',1),pay_key=('fukusho_chakubarai_key',1),place_payload=('odds_fukusho',336))


def schema(directory):
    if (directory/'place_schema.json').exists():return s.read(directory,'place_schema.json')
    with io.session() as db:
        columns=local.table_info(db,'apd_sokuho_o1',local.Budget(120))
        if columns is None:raise safety.MLFailure('invalid_schema')
    value={'table':'public.apd_sokuho_o1','columns':{name:{k:v for k,v in info.items() if k in ('data_type','type_name','type_is_builtin','max_length')}
        for name,info in columns.items()},'source':'read_only_pg_catalog','rows_fetched':0}
    s.write(directory,'place_schema.json',value);return value


def mapping(directory):
    info=schema(directory)
    for field,width in FIELDS.values():
        col=info['columns'].get(field,{})
        if col.get('type_is_builtin') is not True or col.get('type_name') not in local.TEXT_TYPES or col.get('data_type')!=f'character varying({width})':
            raise safety.MLFailure('invalid_schema')
    if s.sha(SPEC,c.ROOT.parent)!=SPEC_SHA:raise safety.MLFailure('mapping_not_reviewed')
    return {'table':info['table'],'columns':{alias:{'physical':name,'width':width} for alias,(name,width) in FIELDS.items()},
        'official_PDF':str(SPEC),'official_PDF_sha256':SPEC_SHA,'PDF_physical_pages':[15,33],'PDF_printed_pages':[12,29],
        'layout':'28_slots_of_12_ASCII;horse2_lower4_upper4_popularity2','price_unit':'integer_divided_by_10_decimal_odds',
        'pay_key':{'0':'not_sold','2':'up_to_second_place','3':'up_to_third_place'},
        'never_infer_pay_key_from_final_or_declared_runner_count':True,'same_formula_for_2_and_3_price_regimes_not_probability':True,
        'missing_tokens':['0000','----','****','    '],'capped_or_ambiguous_tokens':['9999','0999'],
        'cap_note':'post_2004_limit_999.9;legacy_0999_description_ambiguous_conservatively_missing_not_exact_price',
        'record':'same_complete_race_key_and_published_minute_as_saved_T10;status1;whole_win_roster_and_prices_match',
        'original_PC_receipt_and_schedule_asof_still_unverified':True}


def query(races,packets):
    if not 1<=len(races)<=32 or len(set(races))!=len(races):raise safety.MLFailure('input_limit')
    ident=safety.identifier;values=[];params=[]
    for race in races:
        d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));c.guard(d,d)
        q=packets[race]['quotes']['10'];stamp=core.instant(q['published_at'])
        if stamp.year!=d.year:raise safety.MLFailure('invalid_date')
        values.append('('+','.join(['%s::text']*7)+')');params.extend((*race,stamp.strftime('%m%d%H%M')))
    keys=','.join(ident(k) for k in RACE_KEY);equality=' AND '.join('t.'+ident(k)+'::text=w.'+ident(k) for k in RACE_KEY)
    columns=['w.'+ident(k)+' AS '+ident(k) for k in RACE_KEY]
    for alias,(field,width) in FIELDS.items():columns.append(f'CASE WHEN pg_catalog.octet_length(t.{ident(field)})<={width} THEN t.{ident(field)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('WITH w('+keys+',published) AS (VALUES '+','.join(values)+') SELECT '+','.join(columns)+
        ' FROM w LEFT JOIN ONLY public.apd_sokuho_o1 t ON '+equality+' AND t.happyo_tsukihi_jifun::text=w.published')
    validate_audit_sql(sql);return sql,tuple(params)


def numeric(payload,roster):
    if not isinstance(payload,str) or len(payload)!=336 or not payload.isascii():return None,'place_payload_invalid'
    odds={};roster=set(roster)
    for i in range(28):
        slot=payload[12*i:12*(i+1)];h,low,high,pop=slot[:2],slot[2:6],slot[6:10],slot[10:]
        if slot==' '*12:continue
        if h!=f'{i+1:02d}':return None,'place_slot_horse_invalid'
        if low==high and low in ('----','****') and h not in roster:
            if pop!=('--' if low=='----' else '**'):return None,'place_cancel_fields_inconsistent'
            continue
        if low in ('9999','0999') or high in ('9999','0999'):return None,'place_capped_or_ambiguous'
        if any(v in ('0000','----','****','    ') for v in (low,high)):return None,'place_nonprice_or_unavailable'
        if not re.fullmatch('[0-9]{4}',low) or not re.fullmatch('[0-9]{4}',high) or not re.fullmatch('[0-9]{2}',pop) or not 1<=int(pop)<=28:
            return None,'place_fields_invalid'
        l,u=int(low)/10,int(high)/10
        if not 0<l<=u:return None,'place_bounds_invalid'
        odds[h]=(l,u)
    if set(odds)!=roster:return None,'place_roster_mismatch'
    return odds,None


def decode(race,rows,roster,packet):
    if len(rows)!=1:return None,'same_O1_missing' if not rows else 'same_O1_duplicate',None
    row=rows[0];pay_key=row.get('pay_key')
    if row.get('record_type')!='O1' or row.get('status')!='1':return None,'same_O1_category_unconfirmed',pay_key
    reference=packet['quotes']['10'];start=core.instant(packet['scheduled_start'])
    sc=core.Schedule(race,start,None,tuple(roster),True,'reviewed_historical_schedule')
    try:
        checked,why=stage1_analysis.select_quote(race,[dict(row,race_key=race,source_table='apd_sokuho_o1',oversized=False)],roster,sc,10,300)
        if checked is None:return None,why,pay_key
        saved=stage1._quote(reference)
        if checked.published_at!=saved.published_at or checked.odds!=saved.odds:return None,'saved_T10_win_record_changed',pay_key
    except core.SimulationError:return None,'same_O1_time_or_win_invalid',pay_key
    if row.get('place_sale')=='0' and pay_key=='0':return None,'place_not_sold',pay_key
    if row.get('place_sale')!='7' or pay_key not in ('2','3'):return None,'place_sale_or_pay_key_unconfirmed',pay_key
    bounds,why=numeric(row.get('place_payload'),roster)
    if bounds is None:return None,why,pay_key
    values={h:{c.EXTRA[0]:math.log(float(saved.odds[h]))-(math.log(l)+math.log(u))/2,
               c.EXTRA[1]:math.log(u)-math.log(l),c.EXTRA[2]:0} for h,(l,u) in bounds.items()}
    if not all(math.isfinite(v) for row in values.values() for v in row.values()):return None,'place_features_nonfinite',pay_key
    return values,None,pay_key


_raw_cache={}
def inputs(directory,start,end):
    c.guard(start,end);records={}
    for year in range(start.year,end.year+1):
        key=(str(directory),year)
        if key not in _raw_cache:
            result={}
            for path in sorted((directory/'place').glob(f'{year}-*.local.json')):
                doc=s.read(path.parent,path.name)
                for row in doc['records']:
                    race=tuple(row['race_key']);d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));c.guard(d,d)
                    if race in result:raise safety.MLFailure('duplicate_entry')
                    result[race]=row
            _raw_cache[key]=result
        for race,row in _raw_cache[key].items():
            day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
            if start<=day<=end:records[race]=row
    return records


def augment(directory,frame):
    old_data.dates(frame);start,end=date.fromisoformat(str(frame.date.min())),date.fromisoformat(str(frame.date.max()))
    packets=old_data.market(c.S4,start,end);raw=inputs(directory,start,end);values={};stats=Counter();regimes=defaultdict(Counter)
    predictions=exp.predictions(frame,frame.market_probability.to_numpy())
    for p in predictions:
        item=raw.get(p.race)
        if item is None:raise safety.MLFailure('artifact_invalid')
        if item.get('read_error'):v=None;reason='place_read_'+item['read_error'];pay=None
        else:v,reason,pay=decode(p.race,item['rows'],p.probabilities,packets[p.race])
        group=pay if pay in ('2','3') else 'unknown_or_not_sold'
        regimes[group]['races']+=1
        if v:
            for h,r in v.items():values[''.join(p.race)+h]=r
            stats['complete_races']+=1;regimes[group]['complete_races']+=1
        else:stats['missing_races']+=1;stats['reason_'+reason]+=1
    result=frame.copy()
    for name in c.EXTRA:result[name]=[values.get(str(k),{}).get(name,1 if name==c.EXTRA[2] else np.nan) for k in frame.entry_key]
    if result.entry_key.tolist()!=frame.entry_key.tolist():raise safety.MLFailure('invalid_identity')
    count=len(predictions)
    summary={'races':count,'runners':len(frame),'complete_races':stats['complete_races'],'missing_races':stats['missing_races'],
        'complete_race_coverage':stats['complete_races']/count if count else 0.,
        'missing_reasons':{k.removeprefix('reason_'):v for k,v in stats.items() if k.startswith('reason_')},
        'pay_regimes':{k:dict(v,coverage=v['complete_races']/v['races']) for k,v in regimes.items()},'same_primary_population':True}
    return result,summary


def collect(directory):
    done=s.completed(directory,'place')
    if done:return done
    reviewed=mapping(directory);s.write(directory,'place_mapping.json',reviewed)
    # Prior feature manifests/plans are finite and contain no equivalent place-price hypothesis.
    prior_specs=[]
    for root in (c.S2,c.S3,c.S4):
        doc=safety.read_json(root/'plan.json',root=root)
        text=json.dumps(doc)
        prior_specs.append({'source':str(root/'plan.json'),'sha256':s.sha(root/'plan.json',root),
            'equivalent_place_feature_declared':any(name in text for name in ('place_relative_support','place_log_range','odds_fukusho'))})
    if any(v['equivalent_place_feature_declared'] for v in prior_specs):
        value={'fit_allowed':False,'reason':'equivalent_place_experiment_already_declared','new_fits':0,'prior_spec_checks':prior_specs}
        s.write(directory,'place_gate.json',value);s.complete(directory,'place',[directory/'place_gate.json',directory/'place_mapping.json'],value);return value
    folder=directory/'place';folder.mkdir(exist_ok=True);files=[];annual=[]
    for year in range(2020,2026):
        start,end=date(year,1,1),min(date(year,12,31),date(2025,9,30))
        f=old_data.market_frame(c.S4,start,end);packets=old_data.market(c.S4,start,end)
        keys=sorted({p.race for p in exp.predictions(f,f.market_probability.to_numpy())});row_count=0
        for offset in range(0,len(keys),32):
            name=f'{year}-{offset//32:03d}.local.json';files.append(folder/name)
            if (folder/name).exists():row_count+=s.read(folder,name)['rows_fetched'];continue
            batch=keys[offset:offset+32];raw=[];error=None
            try:
                with io.session() as db:
                    actual=publication._columns(db,'apd_sokuho_o1',FIELDS,local.Budget(120))
                    if actual is None:raise safety.MLFailure('invalid_schema')
                    sql,params=query(batch,packets);raw=db.aggregate('s5.same_T10_place',sql,params,max_rows=256)
            except SafeAuditError as exc:error=exc.code
            grouped=defaultdict(list)
            for row in raw:
                race=tuple(row[k] for k in RACE_KEY)
                if row.get('record_type') is not None:grouped[race].append({k:v for k,v in row.items() if k not in RACE_KEY})
            records=[{'race_key':race,'rows':grouped[race],'read_error':error} for race in batch]
            s.write(folder,name,{'records':records,'rows_fetched':len(raw),'batch_races':len(batch),'read_only':True,
                'source':'same_saved_T10_key_and_publication_no_alternative_timestamp','max_rows':256,'statement_timeout_ms':60000})
            row_count+=len(raw)
        _raw_cache.pop((str(directory),year),None)
        _,coverage=augment(directory,f);annual.append(dict(year=year,rows_fetched=row_count,**coverage))
        print(json.dumps(dict(status='running',stage='place',year=year,**coverage)),flush=True)
    folds={};eligible=True
    for year in c.YEARS:
        sp=c.split(year);parts={}
        for part in ('train','validation','calibration','evaluation'):
            begin,finish=(sp[part+'_start'],sp[part+'_end']) if part!='evaluation' else (f'{year}-01-01',f'{year}-09-30')
            f=old_data.market_frame(c.S4,date.fromisoformat(begin),date.fromisoformat(finish));aug,coverage=augment(directory,f)
            if part=='train':
                variation={name:{'finite_rows':int(aug[name].notna().sum()),'distinct_values':int(aug[name].nunique())} for name in c.EXTRA[:2]}
                coverage['training_numeric_variation']=variation
                if not any(v['distinct_values']>1 for v in variation.values()):eligible=False;coverage['gate_reason']='training_numeric_all_constant_or_missing'
            if coverage['complete_race_coverage']<.80:eligible=False;coverage['gate_reason']='complete_race_coverage_below_80_percent'
            parts[part]=coverage
        folds[str(year)]=parts
    gate={'fit_allowed':eligible,'reason':None if eligible else 'place_semantics_timing_coverage_or_variation_gate_failed',
        'annual_source_coverage':annual,'folds':folds,'minimum_coverage_each_partition':.80,'prior_spec_checks':prior_specs,
        'label_associations_not_used_for_gate':True,'place_2_and_3_payees_explicitly_separated':True,'new_fit_limit':3,'new_fits_so_far':0,
        'missing_place_does_not_remove_race':True,'original_PC_receipt_and_schedule_asof_unverified':True}
    s.write(directory,'place_gate.json',gate);files.extend([directory/'place_mapping.json',directory/'place_schema.json',directory/'place_gate.json'])
    s.complete(directory,'place',files,gate);return gate
