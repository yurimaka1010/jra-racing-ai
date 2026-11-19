"""A saved-price favorite baseline, separated from the three frozen AI models."""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import json
import re

from . import pedigree_replay_review as r
from .pedigree_popularity import competition_ranks,CANONICAL_VIEW
from .pedigree_forward.timing import quote_valid

NAME='market_favorite'
LABEL='市場1番人気（T−10）'
VERSION='saved_T10_market_favorite_R1_v1'
SETTLED=('settled','refunded')


def selection(record):
    """Only recorded prices enter this function; outcomes cannot affect selection."""
    r.validate_prediction(record)
    valid=quote_valid(record.get('quote_T10'),record['roster'],record['input_cutoff'])
    if not valid:return {'status':'unavailable','reason':'complete_fresh_T10_quote_unavailable','horse_number':None}
    prices=record['quote_T10']['odds'];ranked=competition_ranks(prices,record['roster'])
    if ranked is None:raise r.safety.MLFailure('invalid_odds')
    lowest=min(Decimal(str(v)) for v in prices.values())
    tied=sorted(h for h in record['roster'] if Decimal(str(prices[h]))==lowest)
    horse=tied[0]
    return {'status':'selected','horse_number':horse,'T10_win_odds':float(prices[horse]),
        'T10_popularity_rank':1,'T10_popularity_tied':len(tied)>1,
        'tied_lowest_horse_numbers':tied,'tied_lowest_count':len(tied),
        'tie_break':'smallest_horse_number','selection_uses_results':False}


def unpack_cache(cache,allowed):
    if cache.get('only_target_dates')!=list(r.DAYS):raise r.safety.MLFailure('invalid_input')
    se=defaultdict(list);ra=defaultdict(list);payouts={}
    for field,target in (('SE',se),('RA',ra)):
        for group in cache[field]:
            for row in group:
                key=r.core.key(row['race_key'])
                if key not in allowed:raise r.safety.MLFailure('invalid_identity')
                target[key].append(row)
    for batch in cache['HR']:
        r.check_day(batch['date'])
        for row in batch['HR_rows']:
            if tuple(row['race_key']) not in allowed:raise r.safety.MLFailure('invalid_identity')
        if batch['HR_rows']:
            for o in r.settle_saved.decode_existing_hr(batch['HR_rows'],batch['reviewed_source']):
                if o.race in payouts:raise r.safety.MLFailure('duplicate_entry')
                payouts[o.race]=o
    return se,ra,payouts


def add(report,records,cache,choices=None):
    if report['dates']!=list(r.DAYS) or report['formal_21_week_included'] is not False:
        raise r.safety.MLFailure('invalid_input')
    by_race={tuple(x['race_key']):x for x in records}
    if len(by_race)!=len(records) or set(by_race)!={tuple(d['race_key']) for d in report['details']}:
        raise r.safety.MLFailure('invalid_identity')
    # Fix every market selection before decoding any answer/payout data.
    proposed={key:selection(rec) for key,rec in by_race.items()}
    if choices is not None and proposed!=choices:raise r.safety.MLFailure('artifact_invalid')
    se,ra,payouts=unpack_cache(cache,set(by_race));result=deepcopy(report)
    for detail in result['details']:
        key=tuple(detail['race_key']);rec=by_race[key];pick=proposed[key]
        if detail['P3']!=(pick['status']=='selected'):raise r.safety.MLFailure('artifact_invalid')
        if not detail['P3']:
            detail[NAME]={'status':'unavailable','horse_number':None,'horse_name':None,'probability':None,
                'T10_popularity_rank':None,'T10_popularity_tied':False,'T10_win_odds':None,
                'T10_popularity_band':'unknown','finish':None,'anomaly':None,'winner_predicted_rank':None,
                'ticket':{'status':'no_bet','reason':'odds_unavailable'},'tied_lowest_count':0,
                'tied_lowest_horse_numbers':[],'missing_reason':'complete_fresh_T10_quote_unavailable'}
            continue
        outcome=payouts.get(key);official,issues=r.align_result(rec,se.get(key,[]),ra.get(key,[]),outcome)
        normal=not issues and r.core.outcome_reason(outcome,tuple(rec['roster'])) is None
        horse=pick['horse_number'];matches=[x for x in official if x['horse_number']==horse]
        row=matches[0] if len(matches)==1 else {};rank=row.get('finish_position')
        finish=int(rank) if isinstance(rank,str) and rank.isdigit() and int(rank)>0 else None
        ticket=r.rules.r1_outcome(horse,official,outcome.payouts_yen_per_100 if normal else {},
            payout_state='normal_confirmed' if normal else 'needs_confirmation')
        detail[NAME]={'status':'selected',**pick,'horse_name':(row.get('horse_name') or '').strip() or None,
            'probability':None,'model_id':None,'T10_popularity_band':'fav1','finish':finish,'anomaly':row.get('anomaly'),
            'winner_predicted_rank':None,'ticket':ticket,'official_issues':issues,
            'missing_reason':ticket.get('reason') if ticket['status']=='unsettled' else None}
    scopes={}
    for key in result['scopes']:
        day,venue=key.split('|')
        selected=[d for d in result['details'] if d['P3'] and (day=='all' or d['date']==day) and (venue=='all' or d['venue_code']==venue)]
        old_common=[d for d in selected if all(d['models'][n]['ticket']['status'] in SETTLED for n in r.NAMES)]
        all_common=[d for d in old_common if d[NAME]['ticket']['status'] in SETTLED]
        def financial(rows,name):return r.finance([d[NAME]['ticket'] if name==NAME else d['models'][name]['ticket'] for d in rows])
        old_finance={n:financial(old_common,n) for n in r.NAMES}
        if key=='all|all' and old_finance!=report['finance_common_settled']:
            raise r.safety.MLFailure('artifact_invalid')
        scopes[key]={'P3_races':len(selected),'old_three_common_races':len(old_common),
            'market_tied_lowest_races':sum(d[NAME]['T10_popularity_tied'] for d in selected),
            'market_only_unsettled_on_three_common':len(old_common)-len(all_common),
            'market_missing_name_races':sum(not d[NAME]['horse_name'] for d in selected),
            'finance_P3':financial(selected,NAME),
            'finance_on_three_common':dict(old_finance,**{NAME:financial(old_common,NAME)}),
            'four_common_races':len(all_common),
            'finance_four_common':{n:financial(all_common,n) for n in (*r.NAMES,NAME)}}
    if scopes['all|all']['P3_races']!=report['P3_eligible_races']:raise r.safety.MLFailure('artifact_invalid')
    result[NAME]={'version':VERSION,'label':LABEL,'scope':'P3_saved_T10_prices_only',
        'stake_yen':100,'tie_break':'lowest_price_then_smallest_horse_number','selection_uses_results':False,
        'official_final_odds_used':False,'new_fits':0,'new_DB_queries':0,'formal_21_week_included':False,'scopes':scopes}
    return result


def unchanged_original(old,new):
    value=deepcopy(new);value.pop(NAME,None)
    for d in value['details']:d.pop(NAME,None)
    return value==old


def stage(directory):
    directory=r.safety.local_path(directory,root=r.ROOT)
    records,_=r.load_predictions();commit=r.read(directory/'inputs.commit.json')
    if any(r.sha(Path(p))!=h for p,h in commit['source_sha256'].items()):raise r.safety.MLFailure('artifact_invalid')
    run=r.safety.local_path(directory/'market-favorite'/uuid4().hex,root=r.ROOT);run.mkdir(parents=True)
    choices={tuple(rec['race_key']):selection(rec) for rec in records}
    r.write(run/'market_choices.local.json',{'version':VERSION,'fixed_at':r.frozen.clock().isoformat(),
        'selections':[dict(value,race_key=list(key)) for key,value in sorted(choices.items())],
        'selected_before_answer_file_opened':True,'from_saved_T10_only':True})
    canonical=directory/'views'/CANONICAL_VIEW
    text=(canonical/'比較レポート.html').read_text(encoding='utf-8')
    old=json.loads(re.search(r'<script id="reviewData" type="application/json">(.*?)</script>',text,re.S)[1])['report']
    if NAME in old:raise r.safety.MLFailure('artifact_exists')
    metadata=r.read(canonical/'presentation.json');official_path=directory/'official_results.local.json'
    if r.sha(official_path)!=metadata['official_source_sha256']:raise r.safety.MLFailure('artifact_invalid')
    enriched=add(old,records,r.read(official_path),choices)
    if not unchanged_original(old,enriched):raise r.safety.MLFailure('artifact_invalid')
    r.write(run/'enriched_report.local.json',enriched)
    from .pedigree_replay_report import save
    view=run/'view';view.mkdir();save(view,enriched)
    r.write(run/'revision.json',{'version':VERSION,'created_at':r.frozen.clock().isoformat(),
        'canonical_view':str(canonical),'staged_view':str(view),'original_report_unchanged':True,
        'market_choices_sha256':r.sha(run/'market_choices.local.json'),'official_source_sha256':r.sha(official_path),
        'new_fits':0,'DB_queries':0,'formal_21_week_included':False})
    scope=enriched[NAME]['scopes']['all|all']
    return {'status':'staged','revision_id':run.name,'report_folder':str(view),
        'P3_races':scope['P3_races'],'old_three_common_races':scope['old_three_common_races'],
        'market_tied_lowest_races':scope['market_tied_lowest_races'],
        'market_only_missing_on_three_common':scope['market_only_unsettled_on_three_common'],
        'finance_on_three_common':scope['finance_on_three_common'],'market_finance_P3':scope['finance_P3'],
        'new_fits':0,'new_DB_queries':0}
