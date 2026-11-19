"""Commit forecasts before opening any results source. Only two explicit dates."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import date,datetime,timedelta
from pathlib import Path
import csv,json,os
import numpy as np
import pandas as pd
from ..core import RACE_KEY
from ..db_connection import read_only_session,validate_audit_sql
from ..ml import safety
from ..research import model as h_model,market_data
from ..stage4 import store as old_store
from ..win_sim import core,publication,local,formats,settle_saved
from ..win_sim_cli import connection_config
from .. import stage1,stage1_analysis

BASE=Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
ROOT=BASE/'prediction-duel'
PROJECT=Path(__file__).resolve().parents[3]
DAYS=('2026-10-17','2026-10-18')
LEGACY_ID='1872a2db908b417a9d982b8836377fa6'
CANOPUS_ID='a8178c2b7523499f9463c1743b41c715'
SAVED_MODELS={
    '2026-10-17':{'id':'legacy_v03','model_id':LEGACY_ID,'temperature':2.,'label':'旧V0.3','timing':'朝7時の保存予測'},
    '2026-10-18':{'id':'canopus_best242','model_id':CANOPUS_ID,'temperature':.75,'label':'カノープスAI（best242）','timing':'09:26:31の保存予測'},
}
S2=BASE/'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S5=BASE/'stage5-development/evaluations/598c62b4dbe54142a2abe4a98b435b12'
S6=BASE/'stage6-development/evaluations/719a0d3915194bc59b6e55eba2659576'
SAVED={'2026-10-17':BASE/'trial-20261017/evaluations/9b2dc7a96aa24008a0dae3a0c34f31e2',
       '2026-10-18':BASE/'trial-20261018/evaluations/c32ac7a4a2ae404fa7e4ec4342856ade'}
PAYOUT=local.ROOT/'simulations/b465a4023292420ba075385876d8d71c'
H_FIELDS=market_data.c.MARKET_FEATURES+market_data.c.HISTORY_FEATURES
VENUES={'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}


def now():return datetime.now(core.JST).isoformat()
def read(path):return safety.read_json(path,root=BASE)
def sha(path):return safety.digest(path,root=BASE)
def write(directory,name,value):safety.write_json(directory/name,value,root=ROOT)


def race_day(race):
    race=core.key(race);day=date(int(race[0]),int(race[1][:2]),int(race[1][2:])).isoformat()
    if day not in DAYS or race[2] not in VENUES:raise safety.MLFailure('invalid_split')
    return day


def session():
    if not os.environ.get('PGPASSWORD'):raise safety.MLFailure('needs_authentication')
    return read_only_session(connection_config(60000),os.environ['PGPASSWORD'])


def saved_cards():
    cards={};sources=[];references={}
    for day,folder in SAVED.items():
        run=read(folder/'run.json');path=folder/'predictions.csv'
        if run.get('status')!='ok' or sha(path)!=run.get('predictions_sha256'):raise safety.MLFailure('artifact_invalid')
        references.update({str(p):sha(p) for p in (path,folder/'run.json')})
        spec=SAVED_MODELS[day]
        authorized=run.get('model_id')==spec['model_id'] and run.get('temperature')==spec['temperature']
        sources.append({'date':day,'saved_model_id':run.get('model_id'),'temperature':run.get('temperature'),
            'comparison_id':spec['id'],'label':spec['label'],'used_as_saved_prediction':authorized,
            'status':'saved_model_verified' if authorized else 'expected_saved_model_missing',
            'other_model_only_supplies_pre_race_roster_and_schedule':not authorized})
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                race=tuple(row[k] for k in RACE_KEY)
                if race_day(race)!=day:raise safety.MLFailure('invalid_split')
                horse=core.horse(row['horse_number']);start=core.instant(row['scheduled_start_jst']);predicted=core.instant(row['predicted_at_jst'])
                if predicted>=start:raise safety.MLFailure('forward_cutoff')
                item=cards.setdefault(race,{'race_key':list(race),'date':day,'venue':VENUES[race[2]],'start':start.isoformat(),
                    'race_number':int(race[-1]),'roster':{},'comparison_id':spec['id'],'saved':{'status':'ready' if authorized else 'missing',
                    'reason':None if authorized else 'expected_saved_model_missing','probabilities':{},'model_id':spec['model_id'],
                    'label':spec['label'],'temperature':spec['temperature'],'predicted_at':predicted.isoformat() if authorized else None,
                    'source_sha256':run['predictions_sha256']}})
                if item['start']!=start.isoformat() or horse in item['roster']:raise safety.MLFailure('duplicate_entry')
                item['roster'][horse]={'horse_id':row['horse_id'],'snapshot_at':predicted.isoformat()}
                if authorized:
                    if row['model_id']!=spec['model_id'] or float(row['temperature'])!=spec['temperature']:raise safety.MLFailure('artifact_invalid')
                    p=float(row['win_probability'])
                    if not np.isfinite(p) or not 0<=p<=1:raise safety.MLFailure('invalid_input')
                    item['saved']['probabilities'][horse]=p
        for race,item in cards.items():
            if item['date']!=day:continue
            if authorized and (abs(sum(item['saved']['probabilities'].values())-1)>1e-12 or set(item['saved']['probabilities'])!=set(item['roster'])):
                raise safety.MLFailure('invalid_identity')
    if len(cards)!=72 or Counter(v['date'] for v in cards.values())!=Counter({d:36 for d in DAYS}):raise safety.MLFailure('invalid_identity')
    return cards,sources,references


def query(races,cards):
    if not 1<=len(races)<=24:raise safety.MLFailure('input_limit')
    ident=safety.identifier;values=[];params=[]
    for race in races:
        race_day(race)
        for minutes in (30,10):
            bound=core.instant(cards[race]['start'])-timedelta(minutes=minutes+1)
            values.append('('+','.join(['%s::text']*6+['%s::integer','%s::text'])+')');params.extend((*race,minutes,bound.strftime('%m%d%H%M')))
    keys=','.join(ident(k) for k in RACE_KEY);equal=' AND '.join('s.'+ident(k)+'::text=w.'+ident(k) for k in RACE_KEY)
    selections=[f'w.{ident(k)}' for k in RACE_KEY]+['w.minutes']
    for alias,(physical,width) in publication.O1_FIELDS.items():
        selections.append(f'CASE WHEN pg_catalog.octet_length(t.{ident(physical)})<={width} THEN t.{ident(physical)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('WITH w('+keys+',minutes,bound) AS (VALUES '+','.join(values)+') SELECT '+','.join(selections)+
        ', EXISTS(SELECT 1 FROM ONLY public.apd_sokuho_o1 s WHERE '+equal+
        " AND (s.data_kubun::text NOT IN ('1','2','3','4','5') OR (s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text !~ '^[0-9]{8}$'))) AS unsafe_version "
        'FROM w LEFT JOIN LATERAL (SELECT s.* FROM ONLY public.apd_sokuho_o1 s WHERE '+equal+
        " AND s.data_kubun::text='1' AND s.happyo_tsukihi_jifun::text<=w.bound ORDER BY s.happyo_tsukihi_jifun DESC LIMIT 2) t ON true")
    validate_audit_sql(sql);return sql,tuple(params)


def collect_prices(cards):
    grouped=defaultdict(list);stats=[]
    # This phase queries O1 only: no RA/SE results, HR, or final-odds rows.
    with session() as db:
        columns=local.table_info(db,'apd_sokuho_o1',local.Budget(60))
        if columns is None:return {},[{'reason':'odds_table_missing'}]
        if any(columns.get(p,{}).get('type_is_builtin') is not True or columns[p]['type_name'] not in local.TEXT_TYPES for p,_ in publication.O1_FIELDS.values()):
            raise safety.MLFailure('invalid_schema')
        races=sorted(cards)
        for offset in range(0,len(races),24):
            sql,params=query(races[offset:offset+24],cards);rows=db.aggregate('duel.pre_result_O1',sql,params,max_rows=96)
            for value in rows:
                race=tuple(value[k] for k in RACE_KEY);race_day(race)
                grouped[(race,int(value['minutes']))].append({k:value[k] for k in publication.O1_FIELDS}|{
                    'race_key':list(race),'unsafe_version':value['unsafe_version'],'oversized':False,'source_table':'apd_sokuho_o1'})
            stats.append({'table':'apd_sokuho_o1','requested_races':min(24,len(races)-offset),'returned_rows':len(rows)})
    return grouped,stats


def select_inputs(cards,prices):
    frames=[];packets={};selected=[]
    for race,item in sorted(cards.items()):
        start=core.instant(item['start']);decision=start-timedelta(minutes=10)
        item['H']={'status':'missing','reason':'odds_unavailable','probabilities':{},'decision_at':decision.isoformat(),
            'timing_basis':'JRA_publication_time','actual_PC_receipt_verified':False,'schedule_basis':'saved_pre_race_card_clock',
            'later_schedule_changes_asof_verified':False}
        if any(core.instant(v['snapshot_at'])>decision for v in item['roster'].values()):item['H']['reason']='roster_snapshot_after_cutoff';continue
        roster=tuple(sorted(item['roster']))
        # The saved CSV proves that this clock/roster was known by prediction time.
        # Supply that actual saved timestamp; do not weaken the shared source gate.
        observed=max(core.instant(v['snapshot_at']) for v in item['roster'].values())
        schedule=core.Schedule(race,start,observed,roster,True,'saved_pre_race_roster_and_clock')
        quotes={};failures={}
        for minutes in (10,30):
            rows=prices.get((race,minutes),[])
            if any(r['unsafe_version'] for r in rows):q,reason=None,'version_unverified'
            else:q,reason=stage1_analysis.select_quote(race,rows,roster,schedule,minutes,300)
            quotes[str(minutes)]=stage1._quote_document(q);failures[str(minutes)]=reason
        selected.append({'race_key':list(race),'scheduled_start':item['start'],'quotes':quotes,'failures':failures,
            'raw_selected_O1':{str(m):prices.get((race,m),[]) for m in (10,30)}})
        if quotes['10'] is None:item['H']['reason']=failures['10'];continue
        packets[race]={'quotes':quotes}
        f=pd.DataFrame({'entry_key':[''.join(race)+h for h in roster],'date':item['date']});frames.append(f)
        item['H'].update(status='ready',reason=None,published_at=quotes['10']['published_at'],early_published_at=quotes['30']['published_at'] if quotes['30'] else None,
            quote_age_seconds=(decision-core.instant(quotes['10']['published_at'])).total_seconds(),history_missing=quotes['30'] is None)
    return market_data.augment(pd.concat(frames,ignore_index=True),packets) if frames else pd.DataFrame(),selected


def parent_directory(run_id):
    import re
    if not re.fullmatch('[a-f0-9]{32}',run_id):raise safety.MLFailure('invalid_input')
    return safety.local_path(ROOT/'evaluations'/run_id,root=ROOT)


def freeze(directory,parent=None):
    if (directory/'forecast_commit.json').exists():return verify_commit(directory)
    cards,sources,refs=saved_cards()
    if parent is not None:
        # Reuse the already-frozen H predictions without opening any results file.
        old_commit=verify_commit(parent);old=read(parent/'forecasts.local.json')
        prior={tuple(v['race_key']):v for v in old['races']}
        if set(prior)!=set(cards) or old.get('results_accessed') is not False:raise safety.MLFailure('artifact_invalid')
        for race,item in cards.items():
            previous=prior[race]
            if item['start']!=previous['start'] or item['roster']!=previous['roster']:raise safety.MLFailure('invalid_identity')
            item['H']=deepcopy(previous['H'])
        refs.update(old['references'])
        for name in ('forecasts.local.json','forecast_commit.json','H_inputs.local.json','dashboard_manifest.json','予想対決.html'):
            refs[str(parent/name)]=sha(parent/name)
        write(directory,'H_inputs.local.json',read(parent/'H_inputs.local.json'))
        forecast={'kind':'saved_best242_and_separate_legacy_V03_vs_frozen_H','schema_version':2,'dates':list(DAYS),
            'created_at':now(),'races':list(cards.values()),'saved_sources':sources,'saved_models':SAVED_MODELS,
            'H_source':old['H_source'],'references':refs,'tasks_before':old_store.task_configuration(),
            'H_input_sha256':sha(directory/'H_inputs.local.json'),'2026_Jan_Sep_read':False,'results_accessed':False,
            'reuse_parent_id':parent.name,'parent_forecast_sha256':old_commit['forecast_sha256'],
            'forecast_interpretation':'unchanged_saved_probabilities_and_prior_result_blinded_H_replay'}
        write(directory,'forecasts.local.json',forecast)
        write(directory,'forecast_commit.json',{'forecast_sha256':sha(directory/'forecasts.local.json'),
            'H_inputs_sha256':sha(directory/'H_inputs.local.json'),'committed_at':now(),
            'results_not_opened_before_commit':True,'reference_hashes':refs})
        return verify_commit(directory)
    trial=read(S2/'trials/2025_annual_market_control.json');folder=S2/'models'/trial['model_id'];bundle=h_model.load(folder,root=S2)
    meta=bundle['metadata'];spec=meta['feature_spec']
    if (tuple(spec['features'])!=H_FIELDS or spec['include_ability'] or meta['calibration']['temperature']!=1.
            or meta['split']['calibration_end']>='2026-01-01'):raise safety.MLFailure('invalid_model')
    refs.update({str(p):sha(p) for p in (folder/'model.json',folder/'win.lgb.txt',S6/'candidate_freeze.json',S5/'development_report.json')})
    for name in ('plan.json','candidate_freeze.json','final_evaluation_plan.json'):
        p=S2/name
        if p.exists():refs[str(p)]=sha(p)
    # Already-confirmed raw O1 format/scale; no interactive re-confirmation or outcome read.
    mapping=read(S5/'place_mapping.json')
    refs[str(S5/'place_mapping.json')]=sha(S5/'place_mapping.json')
    grouped,db_summary=collect_prices(cards);f,selected=select_inputs(cards,grouped)
    if len(f):
        probabilities=h_model.predict(bundle,f)
        for key,p in zip(f.entry_key,probabilities):
            race=(key[:4],key[4:8],key[8:10],key[10:12],key[12:14],key[14:16]);cards[race]['H']['probabilities'][key[16:]]=float(p)
    for item in cards.values():
        item['H'].update(model_id=trial['model_id'],temperature=1.,generated_at=now())
        if item['H']['status']=='ready' and abs(sum(item['H']['probabilities'].values())-1)>1e-12:raise safety.MLFailure('invalid_input')
    write(directory,'H_inputs.local.json',{'records':selected,'input_columns':list(H_FIELDS),'result_columns_read':False,'db_summary':db_summary})
    forecast={'kind':'saved_best242_and_separate_legacy_V03_vs_frozen_H','schema_version':2,
        'dates':list(DAYS),'created_at':now(),'races':list(cards.values()),'saved_sources':sources,'saved_models':SAVED_MODELS,
        'H_source':{'model_id':trial['model_id'],'manifest_sha256':sha(folder/'model.json'),'split':meta['split'],'temperature':1.,
            'features':list(H_FIELDS),'model_retrained':False},'references':refs,'tasks_before':old_store.task_configuration(),
        'H_input_sha256':sha(directory/'H_inputs.local.json'),'2026_Jan_Sep_read':False,'results_accessed':False,
        'forecast_interpretation':'saved_forecasts_and_result_blinded_posthoc_H_replay_not_strict_forward_validation'}
    write(directory,'forecasts.local.json',forecast)
    write(directory,'forecast_commit.json',{'forecast_sha256':sha(directory/'forecasts.local.json'),'H_inputs_sha256':sha(directory/'H_inputs.local.json'),
        'committed_at':now(),'results_not_opened_before_commit':True,'reference_hashes':refs})
    return verify_commit(directory)


def verify_commit(directory):
    commit=read(directory/'forecast_commit.json')
    if sha(directory/'forecasts.local.json')!=commit['forecast_sha256'] or sha(directory/'H_inputs.local.json')!=commit['H_inputs_sha256']:
        raise safety.MLFailure('artifact_invalid')
    for p,digest in commit['reference_hashes'].items():
        if sha(Path(p))!=digest:raise safety.MLFailure('artifact_invalid')
    return commit


def results(directory):
    commit=verify_commit(directory)  # Must happen before opening a result source or database.
    forecast=read(directory/'forecasts.local.json');cards={tuple(v['race_key']):v for v in forecast['races']}
    if (directory/'results.local.json').exists():return read(directory/'results.local.json')
    if forecast.get('reuse_parent_id'):
        # The new forecasts are committed before opening the stored answer key.
        parent=parent_directory(forecast['reuse_parent_id']);old_commit=verify_commit(parent)
        manifest=read(parent/'dashboard_manifest.json')
        if (old_commit['forecast_sha256']!=forecast['parent_forecast_sha256']
                or sha(parent/'results.local.json')!=manifest['result_sha256']):raise safety.MLFailure('artifact_invalid')
        previous=read(parent/'results.local.json')
        if previous['forecast_sha256']!=old_commit['forecast_sha256']:raise safety.MLFailure('artifact_invalid')
        allowed={''.join(race) for race in cards}
        if not set(previous['races'])<=allowed or not set(previous['outcomes'])<=allowed:raise safety.MLFailure('invalid_split')
        data=deepcopy(previous);data.update(forecast_sha256=commit['forecast_sha256'],joined_at=now(),
            reused_from=parent.name,source_result_sha256=manifest['result_sha256'])
        write(directory,'results.local.json',data);return data
    manifest=read(PAYOUT/'manifest.json');profile=read(PAYOUT/'hr_payout_source.json')
    if (sha(PAYOUT/'hr_payout_source.json')!=manifest['provenance']['hr_source_sha256']
            or manifest['provenance'].get('hr_source_confirmed') is not True or profile['numerator']!=1 or profile['denominator']!=1):
        raise safety.MLFailure('mapping_not_reviewed')
    output={};outcomes={};source_status=[]
    with session() as db:
        budget=local.Budget(120);columns=local.table_info(db,'jvd_se',budget)
        fields={'horse_number':('umaban',2),'horse_id':('ketto_toroku_bango',10),'rank':('kakutei_chakujun',2),
            'anomaly':('ijo_kubun_code',1),'status':('data_kubun',1)}
        if columns and 'bamei' in columns:fields['name']=('bamei',256)
        if not columns or any(columns.get(p,{}).get('type_is_builtin') is not True or columns[p]['type_name'] not in local.TEXT_TYPES for p,_ in fields.values()):
            raise safety.MLFailure('invalid_schema')
        for day in DAYS:
            races=sorted(r for r in cards if race_day(r)==day)
            rows=local._rows(db,budget,'jvd_se',fields,columns,races,1500,join_targets=True)
            for row in rows:
                race=core.key(row['race_key']);race_day(race);horse=core.horse(row['horse_number']);output.setdefault(''.join(race),[]).append({k:row.get(k) for k in fields})
            source=deepcopy(profile);source.update(years=[2026],venues=sorted({r[2] for r in races}),race_keys=[list(r) for r in races],allowed_final_categories=['2'],final_category='2')
            hr,source,status=settle_saved.fetch_hr(db,races,max_rows=100,max_seconds=30,reviewed_source=source);source_status.append(status)
            if source:
                for outcome in settle_saved.decode_existing_hr(hr,source):
                    outcomes[''.join(outcome.race)]={'status':outcome.status,'payouts_yen_per_100':outcome.payouts_yen_per_100,
                        'verified':outcome.verified,'complete_winner_set':outcome.complete_winner_set,'reason':outcome.reason}
    data={'races':output,'outcomes':outcomes,'collected_at':now(),'forecast_sha256':commit['forecast_sha256'],'sources':source_status,
        'HR_unit':'previous_owner_confirmed_JPY_per_100_ordinary_win_only','source_profile_sha256':sha(PAYOUT/'hr_payout_source.json')}
    write(directory,'results.local.json',data);return data
