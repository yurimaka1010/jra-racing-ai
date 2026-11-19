"""Reference popularity from already saved T10 quotes. No DB, fit, or inference."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from . import pedigree_replay_review as review
from .pedigree_forward.timing import quote_valid

BANDS=(('fav1','1番人気'),('fav2_3','2〜3番人気'),('fav4_6','4〜6番人気'),('fav7plus','7番人気以下'))
SMALL_SAMPLE=10
VERSION='saved_T10_competition_popularity_v1'
CANONICAL_VIEW='4ea86937f3f24ca8a754eff6ac23126f'


def competition_ranks(prices,roster):
    """1 + number of strictly cheaper horses; a tie gives 1,1,3, not 1,2,3."""
    if not isinstance(prices,dict) or set(prices)!=set(roster) or len(set(roster))!=len(roster):return None
    try:values={h:Decimal(str(prices[h])) for h in roster}
    except (ValueError,TypeError,InvalidOperation):return None
    if any(not v.is_finite() or v<1 for v in values.values()):return None
    ranks={h:1+sum(other<value for other in values.values()) for h,value in values.items()}
    return ranks,{h:sum(other==value for other in values.values())>1 for h,value in values.items()}


def band(rank):
    if type(rank) is not int or rank<1:return 'unknown'
    return 'fav1' if rank==1 else 'fav2_3' if rank<=3 else 'fav4_6' if rank<=6 else 'fav7plus'


def known_hit(detail,model):
    """An unconfirmed finish or a refunded selection is never counted as a loss."""
    if model.get('anomaly') in ('1','2','3') or model['ticket']['status']=='refunded':return None
    ticket=model['ticket']
    if ticket['status']=='settled' and type(ticket.get('hit')) is bool:return ticket['hit']
    if detail.get('results_basis')=='月曜確定区分7':
        if model.get('anomaly') in ('4','5'):return False
        if isinstance(model.get('finish'),int) and model['finish']>0:return model['finish']==1
    return None


def statistics(details):
    common=[d for d in details if d['P3']];out={}
    for name in review.NAMES:
        ranks=[d['models'][name]['T10_popularity_rank'] for d in common if d['models'][name]['T10_popularity_rank'] is not None]
        groups=[]
        for key,label in BANDS:
            selected=[d for d in common if d['models'][name]['T10_popularity_band']==key]
            tickets=[d['models'][name]['ticket'] for d in selected]
            known=[v for d in selected if (v:=known_hit(d,d['models'][name])) is not None]
            settled=[t for t in tickets if t['status'] in ('settled','refunded')]
            cash=review.finance(tickets);part=review.finance(settled);count=len(selected)
            groups.append({'band':key,'label':label,'count':count,
                'share_percent':100*count/len(common) if common else None,
                'known_result_count':len(known),'win_count':sum(known),
                'hit_rate_percent':100*sum(known)/len(known) if known else None,
                'result_unconfirmed_or_refunded_count':count-len(known),
                'finance':cash,'settled_subset':part,
                'small_sample':0<count<SMALL_SAMPLE,
                'note':'対象なし' if not count else '10件未満の少数例。傾向や優劣の判断には使わないでください。' if count<SMALL_SAMPLE else ''})
        out[name]={'P3_races':len(common),'known_popularity_count':len(ranks),
            'unknown_in_P3':len(common)-len(ranks),
            'unknown_outside_P3':sum(d['models'][name]['T10_popularity_rank'] is None for d in details if not d['P3']),
            'mean_popularity_rank':sum(ranks)/len(ranks) if ranks else None,'bands':groups}
    return {'population':'P3','P3_races':len(common),'outside_P3_races':len(details)-len(common),'models':out}


def add(report,records):
    """Return a new report; old metrics, labels, predictions and cohorts stay intact."""
    if report['formal_21_week_included'] is not False or report['dates']!=list(review.DAYS):
        raise review.safety.MLFailure('invalid_input')
    mapped={tuple(r['race_key']):r for r in records}
    if len(mapped)!=len(records) or set(mapped)!={tuple(d['race_key']) for d in report['details']}:
        raise review.safety.MLFailure('invalid_identity')
    enriched=deepcopy(report)
    for detail in enriched['details']:
        record=mapped[tuple(detail['race_key'])];review.validate_prediction(record)
        valid=quote_valid(record.get('quote_T10'),record['roster'],record['input_cutoff'])
        if detail['P3']!=valid:raise review.safety.MLFailure('artifact_invalid')
        quotes=record['quote_T10']['odds'] if valid else {}
        ranked=competition_ranks(quotes,record['roster']) if valid else None
        if valid and ranked is None:raise review.safety.MLFailure('invalid_odds')
        for name in review.NAMES:
            m=detail['models'][name];horse=m['horse_number']
            if (horse!=review.top(record,name) or m['model_id']!=record['models'][name]['model_id']
                    or m['probability']!=record['models'][name]['probability'][horse]):
                raise review.safety.MLFailure('artifact_invalid')
            rank=ranked[0][horse] if ranked else None
            m.update(T10_popularity_rank=rank,T10_popularity_tied=ranked[1][horse] if ranked else False,
                T10_win_odds=float(quotes[horse]) if ranked else None,T10_popularity_band=band(rank),
                T10_popularity_reason=None if ranked else 'complete_fresh_T10_quote_unavailable')
    scopes={}
    for key in enriched['scopes']:
        day,venue=key.split('|')
        selected=[d for d in enriched['details'] if (day=='all' or d['date']==day) and (venue=='all' or d['venue_code']==venue)]
        scopes[key]=statistics(selected)
    enriched['popularity']={'version':VERSION,'source':'saved_T10_win_odds_only',
        'ranking_rule':'competition:1+count(strictly_lower_odds);equal_prices_equal_rank;skip_next_rank',
        'is_official_final_popularity':False,'population':'P3','small_sample_below':SMALL_SAMPLE,
        'scopes':scopes,'band_definitions':[{'key':k,'label':v} for k,v in BANDS],
        'new_fits':0,'new_DB_queries':0,'formal_21_week_included':False}
    if scopes['all|all']['P3_races']!=report['P3_eligible_races']:raise review.safety.MLFailure('artifact_invalid')
    return enriched


def unchanged_original(report,enriched):
    stripped=deepcopy(enriched);stripped.pop('popularity',None)
    for d in stripped['details']:
        for m in d['models'].values():
            for field in tuple(m):
                if field.startswith('T10_popularity_') or field=='T10_win_odds':m.pop(field)
    return stripped==report


def stage(directory):
    directory=review.safety.local_path(directory,root=review.ROOT)
    original=review.read(directory/'report.local.json');records,_=review.load_predictions()
    commit=review.read(directory/'inputs.commit.json')
    if any(review.sha(Path(p))!=h for p,h in commit['source_sha256'].items()):raise review.safety.MLFailure('artifact_invalid')
    enriched=add(original,records)
    if not unchanged_original(original,enriched):raise review.safety.MLFailure('artifact_invalid')
    revision=review.safety.local_path(directory/'popularity'/uuid4().hex,root=review.ROOT)
    revision.mkdir(parents=True,exist_ok=False)
    presentation=review.read(directory/'views'/CANONICAL_VIEW/'presentation.json')
    enriched['result_source_note']=presentation['result_source_note']
    review.write(revision/'enriched_report.local.json',enriched)
    from .pedigree_replay_report import save
    output=revision/'view';output.mkdir();save(output,enriched)
    values=enriched['popularity']['scopes']['all|all']
    review.write(revision/'revision.json',{'version':VERSION,'created_at':review.frozen.clock().isoformat(),
        'source_report_sha256':review.sha(directory/'report.local.json'),
        'original_metrics_and_cohorts_unchanged':True,'source_files_unchanged':len(commit['source_sha256']),
        'canonical_view':str(directory/'views'/CANONICAL_VIEW),'staged_view':str(output),
        'new_fits':0,'DB_queries':0,'formal_21_week_included':False})
    return {'status':'staged','revision_id':revision.name,'report_folder':str(output),
        'P3_races':values['P3_races'],'unknown_outside_P3_races':values['outside_P3_races'],
        'models':{n:{'mean_popularity_rank':m['mean_popularity_rank'],
            'bands':[{'label':b['label'],'count':b['count'],'share_percent':b['share_percent']} for b in m['bands']]}
            for n,m in values['models'].items()},'original_metrics_unchanged':True,'new_fits':0,'new_DB_queries':0}
