"""An explicitly authorized two-day retrospective review, never formal-forward evaluation."""
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import csv
import json
import math
import re

from .core import RACE_KEY
from .ml import safety
from .pedigree_forward import contracts as frozen, evaluation as rules
from .pedigree_forward.timing import quote_valid
from .d39_next.features import race_reasons
from .d39_generation.learning import point
from .win_sim import core, local, settle_saved

ROOT=frozen.BASE/'pedigree-replay-review'
STUDY=frozen.ROOT/'evaluations/c7d80f921b3c42ae910a53a01326f8cb'
SOURCE=STUDY/'replay/a8e9d62efe854490b4e2904eb70df89d'
DAYS=('2026-10-24','2026-10-25')
NAMES=frozen.NAMES
LABELS={'control39':'39列対照','pedigree45':'血統45列','reference47':'休養追加47列','market':'市場確率'}
VENUES={'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
METRICS={'within_race_auc':('レース内AUC',True,'同一レース内での順位識別力'),
         'roc_auc':('全体AUC',True,'全出走馬をまとめた勝馬の識別力'),
         'race_log_loss':('レースLL',False,'勝馬へ割り当てた確率の対数損失'),
         'brier_score':('ブライア得点',False,'馬ごとの確率予測の二乗誤差'),
         'top1_accuracy_percent':('本命1着率',True,'保存順位1位の単勝的中率（％）')}
REASONS={'did_not_finish':'競走中止','scratched':'出走取消','excluded_by_starter':'発走除外',
         'excluded_by_stewards':'競走除外','disqualified':'失格','demoted':'降着','remounted':'再騎乗',
         'anomaly_unknown':'異常区分未確認','incomplete_starter_set':'出走頭数の不整合',
         'tied_or_inconsistent_finish':'同着または着順不整合','invalid_finish':'着順未確定・不正',
         'official_result_missing':'公式結果未収録','official_identity_mismatch':'馬番・血統登録番号不一致',
         'official_category_unconfirmed':'公式結果の区分未確認','official_roster_mismatch':'公式結果の全頭対応不成立',
         'official_race_state_unconfirmed':'レース情報の確定状態未確認',
         'full_order_flash_waiting_for_confirmed_hr':'全着順速報の払戻確定待ち',
         'odds_unavailable':'新鮮で完全なT−10オッズなし','official_hr_unavailable':'通常の公式単勝払戻未確認'}


def write(path,value):safety.write_json(path,value,root=ROOT)


def read(path):return safety.read_json(path,root=frozen.BASE)


def sha(path):return safety.digest(path,root=frozen.BASE)


def check_day(text):
    if text not in DAYS:raise safety.MLFailure('outside_scope')
    return date.fromisoformat(text)


def validate_prediction(record):
    race=core.key(record['race_key']);day=f'{race[0]}-{race[1][:2]}-{race[1][2:]}'
    check_day(day)
    if record.get('date')!=day:raise safety.MLFailure('invalid_identity')
    if (record.get('replay_only') is not True or record.get('H_results_accessed') is not False
            or set(record.get('models',{}))!=set(NAMES)):
        raise safety.MLFailure('outside_scope')
    roster=record['roster']
    if len(set(roster))!=len(roster) or not 2<=len(roster)<=28:raise safety.MLFailure('invalid_identity')
    for n in NAMES:
        m=record['models'][n];p=m.get('probability',{});rank=m.get('rank',{})
        if (m.get('status')!='saved' or m.get('model_id')!=frozen.MODEL_IDS[n]
                or m.get('temperature')!=frozen.TEMPERATURE[n] or set(p)!=set(roster) or set(rank)!=set(roster)
                or not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in p.values())
                or abs(sum(p.values())-1)>1e-12):raise safety.MLFailure('invalid_model')
        expected={h:i for i,h in enumerate(sorted(roster,key=lambda h:(-p[h],h)),1)}
        if rank!=expected:raise safety.MLFailure('artifact_invalid')
    return race


def load_predictions():
    records=[];hashes={};freeze=read(STUDY/'model_freeze.json')
    if sha(STUDY/'model_freeze.json')!=read(STUDY/'model_freeze.commit.json')['sha256']:
        raise safety.MLFailure('artifact_invalid')
    for n,item in freeze['models'].items():
        if n not in NAMES:raise safety.MLFailure('outside_scope')
        for name,field in (('model.json','model_json_sha256'),('booster.lgb.txt','booster_sha256')):
            path=Path(item['model_path'])/name
            if sha(path)!=item[field]:raise safety.MLFailure('artifact_invalid')
            hashes[str(path)]=sha(path)
    for day in DAYS:
        for folder in sorted((SOURCE/day).iterdir()):
            if not folder.is_dir():continue
            receipt=read(folder/'receipt.json');record=read(folder/'prediction.json')
            if receipt.get('status')!='saved' or receipt.get('replay_only') is not True:
                raise safety.MLFailure('artifact_invalid')
            validate_prediction(record)
            for name,field in (('prediction.json','prediction_sha256'),('predictions.local.csv','csv_sha256')):
                path=folder/name
                if sha(path)!=receipt[field]:raise safety.MLFailure('artifact_invalid')
                hashes[str(path)]=sha(path)
            for n in NAMES:
                item=record['models'][n];path=folder/item['input_matrix_file']
                if path.parent!=folder or sha(path)!=item['input_matrix_sha256']:raise safety.MLFailure('artifact_invalid')
                hashes[str(path)]=sha(path)
            records.append(record)
    if len(records)!=72 or sum(len(r['roster']) for r in records)!=935:raise safety.MLFailure('invalid_identity')
    if len({tuple(r['race_key']) for r in records})!=len(records):raise safety.MLFailure('duplicate_entry')
    for name in ('model_freeze.json','model_freeze.commit.json','preregistration.json','preregistration.commit.json'):
        hashes[str(STUDY/name)]=sha(STUDY/name)
    return records,hashes


def checked_fields(db,table,fields,budget):
    columns=local.table_info(db,table,budget)
    if columns is None or any(columns.get(col,{}).get('type_is_builtin') is not True
        or columns[col]['type_name'] not in local.TEXT_TYPES for col,_ in fields.values()):
        raise safety.MLFailure('invalid_schema')
    return columns


def fetch_official(records,session):
    """Exact two-day keys; existing RO query and already reviewed HR decoder only."""
    from .d39_generation.outcomes import reviewed_profile
    grouped=defaultdict(list);races_meta=defaultdict(list);payouts={};sources=[];raw=[]
    for day in DAYS:
        races=tuple(tuple(r['race_key']) for r in records if r['date']==day)
        if len(races)>36:raise safety.MLFailure('input_limit')
        with session() as db:
            budget=local.Budget(30)
            se={'horse_number':('umaban',2),'horse_id':('ketto_toroku_bango',10),
                'finish_position':('kakutei_chakujun',2),'status':('data_kubun',1),
                'anomaly':('ijo_kubun_code',1)}
            cols=checked_fields(db,'jvd_se',se,budget)
            # Display names are optional. Missing or overlong names stay missing.
            col=cols.get('bamei',{})
            if col.get('type_is_builtin') is True and col.get('type_name') in local.TEXT_TYPES:se['horse_name']=('bamei',256)
            rows=local._rows(db,budget,'jvd_se',se,cols,races,1100,join_targets=True)
            ra={'actual_runners':('shusso_tosu',2),'status':('data_kubun',1)}
            columns=checked_fields(db,'jvd_ra',ra,budget)
            race_rows=local._rows(db,budget,'jvd_ra',ra,columns,races,100,join_targets=True)
            try:
                profile=reviewed_profile(races);profile['allowed_final_categories']=['1','2']
                hr,source,status=settle_saved.fetch_hr(db,races,max_rows=100,max_seconds=15,reviewed_source=profile)
                outcomes=settle_saved.decode_existing_hr(hr,source) if hr else []
                for item in outcomes:payouts[item.race]=item
                raw.append({'date':day,'HR_rows':hr,'reviewed_source':source})
            except Exception:
                status={'status':'pending','reason':'official_hr_unavailable'}
            sources.append({'date':day,'SE_rows':len(rows),'RA_rows':len(race_rows),'HR_status':status,
                'normal_confirmed_HR_races':sum(o.status=='normal_final' for o in payouts.values() if o.race in races)})
        for row in rows:grouped[tuple(row['race_key'])].append(row)
        for row in race_rows:races_meta[tuple(row['race_key'])].append(row)
    return {'SE':dict(grouped),'RA':dict(races_meta),'payouts':payouts,'sources':sources,'HR_local':raw}


def align_result(record,rows,race_rows,outcome):
    race=tuple(record['race_key']);expected=set(record['roster']);reasons=[]
    if len(rows)!=len(expected) or {r.get('horse_number') for r in rows}!=expected:
        return [],['official_roster_mismatch']
    if any(r.get('horse_id')!=record['horse_registration_ids'].get(r['horse_number']) for r in rows):
        return [],['official_identity_mismatch']
    if any(r.get('status') not in ('5','6','7') for r in rows):return [],['official_category_unconfirmed']
    if any(not isinstance(r.get('finish_position'),str) or not re.fullmatch('[0-9]{2}',r['finish_position']) for r in rows):
        return [],['invalid_finish']
    if len(race_rows)!=1 or race_rows[0].get('status') not in ('5','6','7'):
        return [],['official_race_state_unconfirmed']
    normal=core.outcome_reason(outcome,tuple(expected)) is None
    winners={r['horse_number'] for r in rows if r.get('finish_position')=='01'}
    if normal and set(outcome.payouts_yen_per_100)!=winners:normal=False;reasons.append('hr_se_winner_mismatch')
    if any(r['status']!='7' for r in rows) and not normal:
        reasons.append('full_order_flash_waiting_for_confirmed_hr')
    result=[dict(r,**dict(zip(RACE_KEY,race)),actual_runners=race_rows[0].get('actual_runners')) for r in rows]
    return result,reasons


def top(record,name):
    ranks=record['models'][name]['rank'];return min(ranks,key=ranks.get)


def metrics(items,name):
    import pandas as pd
    if not items:return None
    rows=[];probability=[];hits=0;ties=0
    for r,result in items:
        p=(rules.market_probability(r,r['roster']) if name=='market' else r['models'][name]['probability'])
        best=min(p,key=lambda h:(-p[h],h)) if name=='market' else top(r,name)
        winner=next(x['horse_number'] for x in result if x['finish_position']=='01')
        hits+=best==winner;ties+=sum(v==max(p.values()) for v in p.values())>1
        for row in sorted(result,key=lambda x:x['horse_number']):
            rows.append({'entry_key':''.join(r['race_key'])+row['horse_number'],'date':r['date'],'winner_label':int(row['finish_position']=='01')})
            probability.append(p[row['horse_number']])
    value=point(pd.DataFrame(rows),probability)
    # The stored recorder breaks ties by horse number, rather than fractional credit.
    value.update(top1_accuracy_percent=100*hits/len(items),top1_hit_count=hits,top_probability_tie_races=ties,
        top_tie_rule='saved_rank;probability_desc_then_horse_number_asc')
    return value


def finance(tickets):
    complete=[x for x in tickets if x['status'] in ('settled','refunded')]
    pending=len(tickets)-len(complete);stake=100*len(tickets);known=sum(x['payout_yen'] for x in complete)
    return {'bet_count':len(tickets),'investment_yen':stake,'hit_count':sum(x.get('hit') is True for x in complete),
        'hit_rate_percent':100*sum(x.get('hit') is True for x in complete)/len(tickets) if tickets and not pending else None,
        'refunded_count':sum(x['status']=='refunded' for x in complete),'unsettled_count':pending,
        'payout_yen':None if pending else known,'net_profit_yen':None if pending else known-stake,
        'return_rate_percent':100*known/stake if stake and not pending else None,
        'settled_count':len(complete),'known_payout_yen':known,'settled_investment_yen':100*len(complete)}


def analyze(records,official):
    p1=[];p2=[];tickets={n:[] for n in NAMES};common_tickets={n:[] for n in NAMES}
    details=[];exclusions=Counter();status_counts=Counter();abnormal=0;odds_missing=0
    for r in records:
        race=tuple(r['race_key']);outcome=official['payouts'].get(race)
        result,issues=align_result(r,official['SE'].get(race,[]),official['RA'].get(race,[]),outcome)
        reason=issues+ (race_reasons(result) if result else [])
        good=not reason;odds_ok=quote_valid(r.get('quote_T10'),r['roster'],r['input_cutoff'])
        odds_missing+=not odds_ok;abnormal+=bool(result and race_reasons(result))
        exclusions.update(set(reason));status_counts.update(row['status'] for row in result)
        if good:p1.append((r,result))
        if good and odds_ok:p2.append((r,result))
        normal=not issues and core.outcome_reason(outcome,tuple(r['roster'])) is None
        by_horse={row['horse_number']:row for row in result}
        item={'date':r['date'],'venue':VENUES.get(race[2],race[2]),'venue_code':race[2],
            'race_no':int(race[5]),'race_key':list(race),'race_id':''.join(race),
            'P1':good,'P2':good and odds_ok,'P3':odds_ok,'odds_ready':odds_ok,
            'exclusion_reasons':sorted(set(reason)),'models':{},'predicted_at':r['prediction_at'],
            'input_cutoff':r['input_cutoff'],'odds_published_at':(r.get('quote_T10') or {}).get('published_at'),
            'odds_first_seen_at':(r.get('quote_T10') or {}).get('first_seen_at'),
            'results_basis':'月曜確定区分7' if result and all(x['status']=='7' for x in result) else '全着順確定速報区分5・6＋通常払戻確認' if not issues else '照合保留'}
        race_tickets={}
        for n in NAMES:
            h=top(r,n);official_row=by_horse.get(h,{});rank=official_row.get('finish_position')
            finish=int(rank) if isinstance(rank,str) and rank.isdigit() and int(rank)>0 else None
            ticket=(rules.r1_outcome(h,result,outcome.payouts_yen_per_100 if normal else {},
                payout_state='normal_confirmed' if normal else 'needs_confirmation') if odds_ok else
                {'status':'no_bet','reason':'odds_unavailable'})
            if odds_ok:tickets[n].append(ticket);race_tickets[n]=ticket
            item['models'][n]={'horse_number':h,'horse_name':(official_row.get('horse_name') or '').strip() or None,
                'probability':r['models'][n]['probability'][h],'finish':finish,'anomaly':official_row.get('anomaly'),
                'model_id':frozen.MODEL_IDS[n],'ticket':ticket,
                'winner_predicted_rank':next((r['models'][n]['rank'][x['horse_number']] for x in result if x['finish_position']=='01'),None)}
        item['top_disagreement']=len({item['models'][n]['horse_number'] for n in NAMES})>1
        if race_tickets and all(v['status'] in ('settled','refunded') for v in race_tickets.values()):
            for n in NAMES:common_tickets[n].append(race_tickets[n])
        details.append(item)
    periods={};scopes={}
    for tag in ('all',*DAYS):
        for venue in ('all',*sorted({r['race_key'][2] for r in records})):
            match=lambda r:(tag=='all' or r['date']==tag) and (venue=='all' or r['race_key'][2]==venue)
            first=[x for x in p1 if match(x[0])];second=[x for x in p2 if match(x[0])]
            selected=[d for d in details if match(d)]
            scope={'P1':{n:metrics(first,n) for n in NAMES},'P2':{n:metrics(second,n) for n in (*NAMES,'market')},
                'P1_races':len(first),'P1_runners':sum(len(x[0]['roster']) for x in first),
                'P2_races':len(second),'P2_runners':sum(len(x[0]['roster']) for x in second),
                'target_races':len(selected),'P3_races':sum(d['P3'] for d in selected),
                'odds_missing_races':sum(not d['odds_ready'] for d in selected),
                'finance':{n:finance([d['models'][n]['ticket'] for d in selected if d['P3']]) for n in NAMES}}
            scopes[tag+'|'+venue]=scope
            if venue=='all':periods[tag]=scope
    base=periods['all']['P1']['control39'];new=periods['all']['P1']['pedigree45']
    delta={k:new[k]-base[k] for k in METRICS} if base and new else {}
    disagreements=[d for d in details if d['P1'] and d['top_disagreement']]
    pairs={}
    for a,b in (('pedigree45','control39'),('pedigree45','reference47'),('control39','reference47')):
        subset=[d for d in details if d['P1'] and d['models'][a]['horse_number']!=d['models'][b]['horse_number']]
        pairs[a+'_vs_'+b]={'different_races':len(subset),
            'first_top_win_count':sum(d['models'][a]['finish']==1 for d in subset),
            'second_top_win_count':sum(d['models'][b]['finish']==1 for d in subset),
            'first_finishes_ahead':sum(d['models'][a]['finish']<d['models'][b]['finish'] for d in subset),
            'second_finishes_ahead':sum(d['models'][b]['finish']<d['models'][a]['finish'] for d in subset)}
    dist={n:dict(Counter(str(d['models'][n]['finish']) for d in details if d['P1'])) for n in NAMES}
    return {'status':'ok','role':'post_event_replay_reference_only_not_forward_or_independent_test','dates':list(DAYS),
        'target_races':len(records),'target_runners':sum(len(r['roster']) for r in records),
        'P1_eligible_races':len(p1),'P2_eligible_races':len(p2),'P3_eligible_races':len(tickets[NAMES[0]]),
        'odds_missing_races':odds_missing,'abnormal_or_inconsistent_races':abnormal,'excluded_P1_races':len(records)-len(p1),
        'exclusion_reasons':dict(exclusions),'SE_category_runner_counts':dict(status_counts),'periods':periods,'scopes':scopes,
        'pedigree_minus_control':delta,'finance_P3':{n:finance(tickets[n]) for n in NAMES},
        'finance_common_settled':{n:finance(common_tickets[n]) for n in NAMES},
        'top_finish_distribution':dist,'top_disagreement_P1_races':len(disagreements),'pair_disagreements':pairs,
        'details':details,'new_fits':0,'formal_21_week_included':False,'H_Stage7_accessed':False,
        'reference_only_no_acceptance_decision':True,'DB_sources':official.get('sources',[])}


def csv_file(path,rows):
    if not rows:return
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def collect(session):
    records,hashes=load_predictions();directory=safety.new_artifact('evaluations',root=ROOT)
    from .stage4.store import task_configuration
    tasks=task_configuration()
    if tasks.get('status')!='observed':raise safety.MLFailure('artifact_invalid')
    write(directory/'inputs.commit.json',{'at':frozen.clock().isoformat(),'source_sha256':hashes,
        'dates':list(DAYS),'new_fits':0,'formal_21_week_included':False,'predictions_verified_before_outcomes':True,
        'tasks':tasks})
    official=fetch_official(records,session)
    write(directory/'official_results.local.json',{'fetched_at':frozen.clock().isoformat(),
        'SE':list(official['SE'].values()),'RA':list(official['RA'].values()),'HR':official['HR_local'],
        'sources':official['sources'],'only_target_dates':list(DAYS)})
    report=analyze(records,official);report.update(artifact_id=directory.name,created_at=frozen.clock().isoformat(),
        source_replay=str(SOURCE),model_ids=frozen.MODEL_IDS,temperatures=frozen.TEMPERATURE,
        caveats=['2026-10-26作成モデルの開催後再生。真正な発走前予測・独立テストではない。',
            '正式な21週間評価へ算入しない。採否判定や温度・モデルの調整に使わない。',
            'オッズ発表時刻と元のPC受信・購入可能時刻は一致すると確認できていない。',
            'T−10表示オッズは払戻を保証しない。収益は確認済みの公式払戻でのみ精算。',
            '血統マスタの過去訂正履歴・当時の完全な発走予定履歴は未確認。'])
    if any(sha(Path(path))!=value for path,value in hashes.items()):raise safety.MLFailure('artifact_invalid')
    report['existing_predictions_models_and_preregistration_unchanged']=True
    if task_configuration()!=tasks:raise safety.MLFailure('artifact_invalid')
    report['existing_tasks_unchanged']=True
    write(directory/'report.local.json',report)
    aggregate={k:v for k,v in report.items() if k not in ('details','scopes')}
    write(directory/'summary.json',aggregate)
    return directory,report


def render(directory):
    directory=safety.local_path(directory,root=ROOT)
    if not directory.is_dir():raise safety.MLFailure('artifact_invalid')
    report=read(directory/'report.local.json')
    if report['formal_21_week_included'] is not False or report['dates']!=list(DAYS):raise safety.MLFailure('outside_scope')
    from .pedigree_popularity import add
    records,_=load_predictions();report=add(report,records)
    from .pedigree_market_favorite import selection,add as add_market
    market_choices={tuple(rec['race_key']):selection(rec) for rec in records}
    original=read(directory/'official_results.local.json')
    report=add_market(report,records,original,market_choices)
    categories=Counter(str(row.get('status')) for batch in original['HR'] for row in batch['HR_rows'])
    se='・'.join(f'{k}（{v}頭）' for k,v in sorted(report['SE_category_runner_counts'].items()))
    hr='・'.join(f'{k}（{v}レース）' for k,v in sorted(categories.items()))
    report['result_source_note']=('使用データ版：着順SE区分'+se+'／払戻HR区分'+hr+'。'
        '区分5・6の全着順確定速報は、既存処理に従い通常払戻を確認できたレースを採用。月曜確定版と同一視していません。')
    target=directory
    if (directory/'比較レポート.html').exists():
        from uuid import uuid4
        target=safety.local_path(directory/'views'/uuid4().hex,root=ROOT);target.mkdir(parents=True,exist_ok=False)
    from .pedigree_replay_report import save
    save(target,report)
    write(target/'presentation.json',{'source_report_sha256':sha(directory/'report.local.json'),
        'official_source_sha256':sha(directory/'official_results.local.json'),'created_at':frozen.clock().isoformat(),
        'html_sha256':sha(target/'比較レポート.html'),'result_source_note':report['result_source_note'],
        'no_model_or_prediction_recalculation':True,'new_fits':0,'formal_21_week_included':False})
    return {'status':'rendered','artifact_id':directory.name,'report_folder':str(target),
        'new_fits':0,'formal_21_week_included':False}


def aggregate_output(directory,report):
    return {'status':'ok','artifact_id':directory.name,'report_folder':str(directory),
        'P1_races':report['P1_eligible_races'],'P2_races':report['P2_eligible_races'],'P3_races':report['P3_eligible_races'],
        'metrics':{n:{k:v for k,v in m.items() if k in METRICS or k in ('races','runners','top1_hit_count')} if m else None
            for n,m in report['periods']['all']['P1'].items()},'pedigree_minus_control':report['pedigree_minus_control'],
        'market_common_metrics':{n:{k:v for k,v in m.items() if k in METRICS} if m else None
            for n,m in report['periods']['all']['P2'].items()},'finance':report['finance_P3'],
        'exclusions':report['exclusion_reasons'],'new_fits':0,'formal_21_week_included':False}
